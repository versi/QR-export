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

# RS parameters (modifiable)
K = 7              # nombre de chunks data par groupe
N = 10             # total chunks par groupe (data + parité)
CHUNK_SIZE = 150   # taille en octets d’un chunk

def sha256(data: bytes) -> str:
    """Calcule le hash SHA-256 d'un buffer."""
    return hashlib.sha256(data).hexdigest()

def split_file_in_chunks(data: bytes, chunk_size: int) -> list:
    """Découpe les données en chunks de taille chunk_size."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def pad_chunk(chunk: bytes, size: int) -> bytes:
    """Pad un chunk pour atteindre la taille size (ajoute des \x00)."""
    return chunk + b'\x00' * (size - len(chunk))

def encode_rs_group(data_chunks: list, n: int, k: int) -> list:
    """
    Encode un groupe de k chunks avec RS pour générer n-k chunks de parité.
    Renvoie une liste de n chunks (k data + n-k parité).
    Chaque chunk est de taille chunk_size (après padding).
    """
    rs = RSCodec(n - k)
    # Concatène les chunks data en un seul buffer
    group_data = b''.join(data_chunks)
    # Encode le groupe entier
    encoded = rs.encode(group_data)
    # Taille de chaque chunk encodé
    chunk_len = len(encoded) // n
    # Découpe en n chunks
    return [encoded[i * chunk_len:(i + 1) * chunk_len] for i in range(n)]

def show_qr_image(img: Image.Image) -> None:
    """Affiche une image QR code avec OpenCV ou PIL."""
    if USE_CV2:
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.imshow("QR Code", img_cv)
        cv2.waitKey(1)
    else:
        img.show()

def encode_file_to_qrs(file_path: str, interval_ms: int) -> None:
    """Encode le fichier en QR codes diffusés à l'écran avec correction d'erreur RS par groupe."""
    with open(file_path, "rb") as f:
        data = f.read()

    filename = os.path.basename(file_path)
    file_hash = sha256(data)

    # Découpe tout le fichier en chunks (non pad pour l’instant)
    all_chunks = split_file_in_chunks(data, CHUNK_SIZE)

    # Nombre total de groupes
    total_groups = (len(all_chunks) + K - 1) // K

    print(f"📡 Diffusion des QR codes - fichier: {filename}")
    print(f"Taille fichier: {len(data)} octets, chunks: {len(all_chunks)}, groupes: {total_groups}")

    # Diffuse groupe par groupe
    qr_index = 0
    for group_idx in range(total_groups):
        # Extraction des chunks du groupe (avec padding si nécessaire)
        start = group_idx * K
        end = start + K
        group_chunks = all_chunks[start:end]
        # Pad les chunks pour qu'ils fassent tous CHUNK_SIZE
        group_chunks = [pad_chunk(c, CHUNK_SIZE) for c in group_chunks]
        # Si moins de k chunks, pad pour arriver à k
        while len(group_chunks) < K:
            group_chunks.append(b'\x00' * CHUNK_SIZE)

        # Encode RS sur le groupe (k data + n-k parité)
        encoded_chunks = encode_rs_group(group_chunks, N, K)

        # Génère et affiche les QR codes pour ce groupe
        for i, chunk in enumerate(encoded_chunks):
            payload = {
                "file_name": filename,
                "file_hash": file_hash,
                "group_index": group_idx,
                "total_groups": total_groups,
                "chunk_index": i,
                "chunks_per_group": N,
                "data_chunks": K,
                "chunk_size": len(chunk),
                "chunk_hash": sha256(chunk),
                "data": base64.b64encode(chunk).decode("utf-8"),
            }

            json_payload = json.dumps(payload)
            qr = qrcode.make(json_payload)
            show_qr_image(qr)

            qr_index += 1
            print(f"[QR {qr_index}] Group {group_idx+1}/{total_groups} chunk {i+1}/{N} affiché")
            time.sleep(interval_ms / 1000.0)

    if USE_CV2:
        print("✅ Diffusion terminée. Appuyez sur une touche pour fermer.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Diffuse un fichier en QR codes avec correction RS(n,k) par groupe."
    )
    parser.add_argument("file", help="Chemin du fichier à encoder")
    parser.add_argument("-i", "--interval", type=int, default=1000, help="Intervalle entre QR en ms")
    args = parser.parse_args()

    encode_file_to_qrs(args.file, args.interval)

if __name__ == "__main__":
    main()
