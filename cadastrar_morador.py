import cv2
import os
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from banco import inserir_morador, conectar


# =========================
# PASTAS
# =========================

os.makedirs("fotos", exist_ok=True)
os.makedirs("faces", exist_ok=True)


# =========================
# MODELOS
# =========================

detector = MTCNN()
embedder = FaceNet()


# =========================
# DADOS
# =========================

nome = input("Nome: ")
apartamento = input("Apartamento: ")
bloco = input("Bloco: ")


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro câmera")
    exit()


print("Espaço = capturar | ESC = sair")

embedding_final = None
frame_final = None


while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face["box"]

        x = max(0, x)
        y = max(0, y)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Cadastro", frame)

    key = cv2.waitKey(1)

    if key == 27:
        print("Cancelado")
        exit()

    if key == 32 and len(faces) > 0:

        x, y, w, h = faces[0]["box"]

        x = max(0, x)
        y = max(0, y)

        rosto = rgb[y:y+h, x:x+w]

        if rosto.size == 0:
            continue

        embedding_final = embedder.embeddings([rosto])[0]
        frame_final = rosto.copy()

        break


cap.release()
cv2.destroyAllWindows()


# =========================
# SALVAR NO BANCO
# =========================

if embedding_final is None:
    print("Nenhum rosto capturado.")
    exit()

morador_id = inserir_morador(
    nome,
    apartamento,
    bloco,
    "",
    ""
)

print("ID:", morador_id)


foto_path = f"fotos/{morador_id}.jpg"
embed_path = f"faces/{morador_id}.npy"

cv2.imwrite(foto_path, frame_final)
np.save(embed_path, embedding_final)


# =========================
# UPDATE SEGURO
# =========================

conn = conectar()
cursor = conn.cursor()

cursor.execute("""
    UPDATE morador
    SET caminho_foto = ?,
        caminho_embedding = ?
    WHERE id = ?
""", (foto_path, embed_path, morador_id))

conn.commit()
conn.close()

print("Cadastro concluído!")
