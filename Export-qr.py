import os
import time
import json
import base64
import hashlib
import argparse
import qrcode
from reedsolo import RSCodec
from PIL import Image
import io

try:
    import cv2
    import numpy as np
    USE_CV2 = True
except ImportError:
    USE_CV2 = False

# Paramètres RS
DATA_CHUNKS = 6
TOTAL_CHUNKS = 10
CHUNK_SIZE = 200  # octets

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def split_and_pad(data, chunk_size, target_chunks):
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    while len(chunks) < target_chunks:
        chunks.append(b'\x00' * chunk_size)
    return chunks

def encode_rs_chunks(chunks, n, k):
    rs = RSCodec(n - k)
    encoded = rs.encode(b''.join(chunks))
    block_size = len(encoded) // n
    return [encoded[i:i+block_size] for i in range(0, len(encoded), block_size)]

def show_qr_image(img):
    if USE_CV2:
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.imshow("QR Code", img_cv)
        cv2.waitKey(1)
    else:
        img.show()

def encode_and_show(file_path, interval_ms):
    with open(file_path, 'rb') as f:
        data = f.read()

    filename = os.path.basename(file_path)
    file_hash = sha256(data)

    chunks = split_and_pad(data, CHUNK_SIZE, DATA_CHUNKS)
    rs_chunks = encode_rs_chunks(chunks, TOTAL_CHUNKS, DATA_CHUNKS)

    print(f"📡 Diffusion des QR codes toutes les {interval_ms} ms (RS({TOTAL_CHUNKS},{DATA_CHUNKS}))")

    for i, chunk in enumerate(rs_chunks):
        payload = {
            "index": i,
            "total": TOTAL_CHUNKS,
            "filename": filename,
            "file_hash": file_hash,
            "chunk_hash": sha256(chunk),
            "rs_k": DATA_CHUNKS,
            "rs_n": TOTAL_CHUNKS,
            "data": base64.b64encode(chunk).decode('utf-8')
        }

        qr = qrcode.make(json.dumps(payload))
        show_qr_image(qr)

        print(f"[{i+1}/{TOTAL_CHUNKS}] QR affiché.")
        time.sleep(interval_ms / 1000.0)

    if USE_CV2:
        print("✅ Fin de la diffusion. Appuyez sur une touche pour fermer la fenêtre.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Afficher des QR codes RS(n,k) à partir d’un fichier avec intervalle en millisecondes.")
    parser.add_argument("file", help="Chemin du fichier à encoder")
    parser.add_argument("-i", "--interval", type=int, default=1000, help="Intervalle entre QR codes (millisecondes)")
    args = parser.parse_args()

    encode_and_show(args.file, args.interval)
