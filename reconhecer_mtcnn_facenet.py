import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

LIMIAR_RECONHECIMENTO = 0.9

MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "condominio/acesso"

# =========================
# MQTT
# =========================

client = mqtt.Client()

client.connect(
    MQTT_HOST,
    MQTT_PORT,
    60
)

# =========================
# MTCNN
# =========================

detector = MTCNN()

# =========================
# FACENET
# =========================

embedder = FaceNet()

embedding_yasmin = np.load(
    "faces/yasmin_embedding.npy"
)

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

ultimo_envio = 0
ultimo_nome = ""

print("Sistema iniciado...")
print("Pressione ESC para sair.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar webcam.")
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    faces = detector.detect_faces(rgb)

    for face in faces:

        try:

            x, y, w, h = face["box"]

            x = max(0, x)
            y = max(0, y)

            rosto = rgb[
                y:y+h,
                x:x+w
            ]

            if rosto.size == 0:
                continue

            embedding_atual = embedder.embeddings(
                [rosto]
            )[0]

            distancia = np.linalg.norm(
                embedding_yasmin -
                embedding_atual
            )

            nome = "Desconhecido"
            autorizado = False
            gate = "closed"

            if distancia < LIMIAR_RECONHECIMENTO:

                nome = "Yasmin"
                autorizado = True
                gate = "open"

            # -----------------
            # Desenha retângulo
            # -----------------

            cor = (0, 0, 255)

            if autorizado:
                cor = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                cor,
                2
            )

            cv2.putText(
                frame,
                f"{nome} ({distancia:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                cor,
                2
            )

            # -----------------
            # MQTT
            # -----------------

            agora = time.time()

            if (
                agora - ultimo_envio > 5
                or nome != ultimo_nome
            ):

                payload = {
                    "name": nome,
                    "authorized": autorizado,
                    "gate": gate,
                    "distance": round(
                        float(distancia),
                        3
                    ),
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }

                client.publish(
                    MQTT_TOPIC,
                    json.dumps(payload)
                )

                print(
                    json.dumps(
                        payload,
                        indent=2
                    )
                )

                ultimo_envio = agora
                ultimo_nome = nome

        except Exception as e:

            print(
                "Erro:",
                e
            )

    cv2.imshow(
        "Controle de Acesso - Condominio",
        frame
    )

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break

# =========================
# FINALIZAÇÃO
# =========================

cap.release()

cv2.destroyAllWindows()

client.disconnect()

print("Sistema encerrado.")