import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import paho.mqtt.client as mqtt
import json
import time
import os
from banco import listar_moradores
from banco import registrar_acesso
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

LIMIAR_RECONHECIMENTO = 0.9

MQTT_HOST = "192.168.64.2"
MQTT_PORT = 1883
MQTT_TOPIC = "condominio/acesso"

# =========================
# MQTT
# =========================

client = mqtt.Client()

client.username_pw_set(
    "mqtt",
    "mqtt123"
)

print("Conectando MQTT...")
print(MQTT_HOST, MQTT_PORT)

client.connect(
    MQTT_HOST,
    MQTT_PORT,
    60
)

print("MQTT conectado!")

client.loop_start()

# =========================
# MTCNN
# =========================

detector = MTCNN()

# =========================
# FACENET
# =========================

embedder = FaceNet()

moradores = {}

for morador in listar_moradores():

    (
        morador_id,
        nome,
        apartamento,
        bloco,
        caminho_foto,
        caminho_embedding,
        ativo
    ) = morador

    if os.path.exists(caminho_embedding):
        moradores[morador_id] = {
            "nome": nome,
            "apartamento": apartamento,
            "bloco": bloco,
            "embedding": np.load(
                caminho_embedding
            )
        }

print(
    f"{len(moradores)} moradores carregados."
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

            menor_distancia = 999
            melhor_id = None

            for morador_id, dados in moradores.items():

                distancia = np.linalg.norm(
                    dados["embedding"] -
                    embedding_atual
                )

                if distancia < menor_distancia:

                    menor_distancia = distancia
                    melhor_id = morador_id

            nome = "Desconhecido"
            apartamento = ""
            bloco = ""

            autorizado = False
            gate = "closed"

            if (
                melhor_id is not None
                and menor_distancia <
                LIMIAR_RECONHECIMENTO
            ):

                dados = moradores[melhor_id]

                nome = dados["nome"]
                apartamento = dados["apartamento"]
                bloco = dados["bloco"]

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
                f"{nome} ({menor_distancia:.2f})",
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
                    "morador_id": (
                        melhor_id
                        if autorizado
                        else None
                    ),
                    "nome": nome,
                    "apartamento": apartamento,
                    "bloco": bloco,
                    "authorized": autorizado,
                    "gate": gate,
                    "distance": round(
                        float(menor_distancia),
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
                
                if autorizado:
                    registrar_acesso(
                        morador_id=melhor_id,
                        autorizado=1,
                        distancia=float(
                            menor_distancia
                        ),
                        data_hora=datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
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
