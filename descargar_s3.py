import argparse
import csv
import datetime
import hashlib
import hmac
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SHA256_VACIO = hashlib.sha256(b"").hexdigest()


def llaves(ruta_csv):
    if ruta_csv:
        with open(ruta_csv, encoding="utf-8-sig", newline="") as f:
            fila = {k.strip().lower(): (v or "").strip() for k, v in next(csv.DictReader(f)).items() if k}
        return fila.get("access key id"), fila.get("secret access key")
    llave, secreto = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not llave or not secreto:
        raise SystemExit("Faltan AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY (o --csv).")
    return llave, secreto


def _hmac(clave, mensaje):
    return hmac.new(clave, mensaje.encode(), hashlib.sha256).digest()


def _get(cred, host, ruta, region, parametros=None):
    llave, secreto = cred
    ahora = datetime.datetime.now(datetime.timezone.utc)
    fecha, dia = ahora.strftime("%Y%m%dT%H%M%SZ"), ahora.strftime("%Y%m%d")
    consulta = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
        for k, v in sorted((parametros or {}).items())
    )
    ruta = urllib.parse.quote(ruta, safe="/-_.~")
    firmadas = "host;x-amz-content-sha256;x-amz-date"
    canonica = (
        f"GET\n{ruta}\n{consulta}\nhost:{host}\nx-amz-content-sha256:{SHA256_VACIO}\n"
        f"x-amz-date:{fecha}\n\n{firmadas}\n{SHA256_VACIO}"
    )
    alcance = f"{dia}/{region}/s3/aws4_request"
    a_firmar = f"AWS4-HMAC-SHA256\n{fecha}\n{alcance}\n{hashlib.sha256(canonica.encode()).hexdigest()}"
    clave = _hmac(("AWS4" + secreto).encode(), dia)
    for parte in (region, "s3", "aws4_request"):
        clave = _hmac(clave, parte)
    firma = hmac.new(clave, a_firmar.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"https://{host}{ruta}" + (f"?{consulta}" if consulta else ""))
    req.add_header("x-amz-date", fecha)
    req.add_header("x-amz-content-sha256", SHA256_VACIO)
    req.add_header(
        "Authorization",
        f"AWS4-HMAC-SHA256 Credential={llave}/{alcance}, SignedHeaders={firmadas}, Signature={firma}",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get(cred, host, ruta, region, parametros=None, intentos=6):
    for intento in range(1, intentos + 1):
        try:
            status, cuerpo = _get(cred, host, ruta, region, parametros)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if intento == intentos:
                raise
            time.sleep(2**intento)
            continue
        if status < 500 or intento == intentos:
            return status, cuerpo
        time.sleep(2**intento)


def error_aws(cuerpo):
    try:
        raiz = ET.fromstring(cuerpo)
        return f"{raiz.findtext('Code')}: {raiz.findtext('Message')}"
    except ET.ParseError:
        return cuerpo[:200].decode("utf-8", "replace")


def region(bucket):
    req = urllib.request.Request(f"https://{bucket}.s3.amazonaws.com/", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.headers.get("x-amz-bucket-region", "us-east-1")
    except urllib.error.HTTPError as e:
        return e.headers.get("x-amz-bucket-region", "us-east-1")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return "us-east-1"


def listar(cred, host, region_, prefijo):
    objetos, token = [], None
    while True:
        parametros = {"list-type": "2", "prefix": prefijo}
        if token:
            parametros["continuation-token"] = token
        status, cuerpo = get(cred, host, "/", region_, parametros)
        if status != 200:
            raise SystemExit(f"Error listando {prefijo} (HTTP {status}) -> {error_aws(cuerpo)}")
        raiz = ET.fromstring(cuerpo)
        for obj in raiz.iterfind("s3:Contents", NS):
            clave = obj.findtext("s3:Key", namespaces=NS)
            if not clave.endswith("/"):
                objetos.append((clave, int(obj.findtext("s3:Size", namespaces=NS))))
        if raiz.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return objetos
        token = raiz.findtext("s3:NextContinuationToken", namespaces=NS)


def descargar(cred, host, region_, clave, tam, destino):
    ruta = destino / clave
    if ruta.exists() and ruta.stat().st_size == tam:
        return "omitido"
    try:
        status, cuerpo = get(cred, host, "/" + clave, region_)
        if status != 200:
            return f"error {clave} (HTTP {status}) -> {error_aws(cuerpo)}"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_name(ruta.name + ".tmp")
        temporal.write_bytes(cuerpo)
        os.replace(temporal, ruta)
    except Exception as e:
        return f"error {clave} -> {e}"
    return "descargado"


def main():
    p = argparse.ArgumentParser(description="Descarga prefijos de un bucket S3 conservando la ruta de cada objeto.")
    p.add_argument("bucket")
    p.add_argument("prefijos", nargs="+", help="Ej. dataset/cpt/sentencias/ags/")
    p.add_argument("--destino", default=".")
    p.add_argument("--csv", default=None, help="CSV de llaves de IAM; sin esto se usan las variables de entorno.")
    p.add_argument("--max-archivos", type=int, default=None, help="Por prefijo; para probar.")
    p.add_argument("--hilos", type=int, default=16)
    a = p.parse_args()

    cred = llaves(a.csv)
    region_ = region(a.bucket)
    host = f"{a.bucket}.s3.{region_}.amazonaws.com"
    destino = Path(a.destino)

    objetos = []
    for prefijo in a.prefijos:
        encontrados = listar(cred, host, region_, prefijo)[: a.max_archivos]
        print(f"{prefijo}: {len(encontrados)} objetos, {sum(t for _, t in encontrados) / 1e6:.1f} MB")
        objetos += encontrados

    conteo = {"descargado": 0, "omitido": 0}
    errores = []
    with ThreadPoolExecutor(a.hilos) as ex:
        for i, r in enumerate(ex.map(lambda o: descargar(cred, host, region_, *o, destino), objetos), 1):
            if r in conteo:
                conteo[r] += 1
            else:
                errores.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(objetos)}", file=sys.stderr)
    print(f"descargados={conteo['descargado']} omitidos={conteo['omitido']} errores={len(errores)}")
    for e in errores[:20]:
        print(f"  {e}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
