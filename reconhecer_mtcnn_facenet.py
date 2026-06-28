import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import paho.mqtt.client as mqtt
import json
import time
import os
import onnxruntime as ort
from banco import listar_moradores, registrar_acesso
from datetime import datetime
import platform

# =========================
# CONFIGURAÇÕES
# =========================

LIMIAR_RECONHECIMENTO = 0.9
LIMIAR_ANTI_SPOOFING = 0.6

MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_ENVIO = "condominio/acesso"
MQTT_TOPIC_STATUS = "condominio/portao/status"

portao_status = "idle"
cooldowns_moradores = {}
ultimo_envio = 0

# =========================
# MQTT
# =========================

def on_message(client, userdata, msg):
    global portao_status
    if msg.topic == MQTT_TOPIC_STATUS:
        portao_status = msg.payload.decode().strip()
        print(f"[MQTT] Status do portão: {portao_status}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set("mqtt", "mqtt123")
client.on_message = on_message

print("Conectando ao MQTT...")
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_STATUS)
client.loop_start()

# =========================
# MODELOS
# =========================

detector = MTCNN()
embedder = FaceNet()

caminho_onnx = "modelos/MiniFASNetV2.onnx"
if not os.path.exists(caminho_onnx):
    raise FileNotFoundError(f"Coloque o modelo em: {caminho_onnx}")

ort_session = ort.InferenceSession(caminho_onnx, providers=['CPUExecutionProvider'])
nome_entrada = ort_session.get_inputs()[0].name

# =========================
# MORADORES
# =========================

moradores = {}

for morador in listar_moradores():
    morador_id, nome_morador, apartamento, bloco, _, caminho_embedding, _ = morador

    if os.path.exists(caminho_embedding):
        moradores[morador_id] = {
            "nome": nome_morador,
            "apartamento": apartamento,
            "bloco": bloco,
            "embedding": np.load(caminho_embedding)
        }

print(f"{len(moradores)} moradores carregados.")

# =========================
# ANTI-SPOOFING
# =========================

def verificar_vivacidade(rosto_img):
    try:
        rosto = cv2.resize(rosto_img, (80, 80))
        rosto = rosto.astype(np.float32)
        rosto = np.transpose(rosto, (2, 0, 1))
        rosto = np.expand_dims(rosto, axis=0)

        saida = ort_session.run(None, {nome_entrada: rosto})[0]

        exp = np.exp(saida - np.max(saida))
        prob = exp / exp.sum()

        return float(prob[0][1])

    except Exception as e:
        print("Erro Anti-Spoofing:", e)
        return 0.0

# =========================
# CÂMERA (CROSS-PLATFORM)
# =========================

def abrir_camera(index=0):
    sistema = platform.system().lower()

    if "windows" in sistema:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif "darwin" in sistema:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)

    return cap


cap = abrir_camera(0)

if not cap.isOpened():
    print("ERRO: câmera não abriu")
    exit()

print("Sistema iniciado com Anti-Spoofing. Pressione ESC para sair.")

# =========================
# LOOP PRINCIPAL
# =========================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar frame")
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for face in faces:
        try:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)
            rosto = rgb[y:y+h, x:x+w]

            if rosto.size == 0:
                continue

            score_real = verificar_vivacidade(rosto)
            eh_real = score_real > LIMIAR_ANTI_SPOOFING

            nome = "Desconhecido"
            apartamento, bloco = "", ""
            autorizado = False
            agora = time.time()

            if eh_real:
                embedding_atual = embedder.embeddings([rosto])[0]

                menor_distancia = 999
                melhor_id = None

                for morador_id, dados in moradores.items():
                    distancia = np.linalg.norm(dados["embedding"] - embedding_atual)

                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        melhor_id = morador_id

                if melhor_id is not None and menor_distancia < LIMIAR_RECONHECIMENTO:
                    dados = moradores[melhor_id]
                    nome = dados["nome"]
                    apartamento = dados["apartamento"]
                    bloco = dados["bloco"]
                    autorizado = True
            else:
                print(f"[FRAUDE] Score: {score_real:.2f}")

            # =========================
            # UI
            # =========================

            if not eh_real:
                cor = (0, 0, 255)
                status_txt = f"FOTO DETECTADA ({score_real:.2f})"
            elif autorizado:
                if portao_status != "idle":
                    cor = (255, 165, 0)
                    status_txt = "Portao Ocupado"
                else:
                    cor = (0, 255, 0)
                    status_txt = f"Real: {nome}"
            else:
                cor = (0, 0, 255)
                status_txt = "Desconhecido"

            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(frame, status_txt, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            # =========================
            # MQTT
            # =========================

            if autorizado and eh_real and portao_status == "idle":

                if melhor_id not in cooldowns_moradores or (agora - cooldowns_moradores[melhor_id] > 30):

                    payload = {
                        "morador_id": melhor_id,
                        "nome": nome,
                        "apartamento": apartamento,
                        "bloco": bloco,
                        "authorized": True,
                        "gate": "open",
                        "distance": round(float(menor_distancia), 3),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    client.publish(MQTT_TOPIC_ENVIO, json.dumps(payload))

                    registrar_acesso(
                        morador_id=melhor_id,
                        autorizado=1,
                        distancia=float(menor_distancia),
                        data_hora=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )

                    print(f"[OK] Acesso liberado para {nome}")
                    cooldowns_moradores[melhor_id] = agora

            elif not eh_real and agora - ultimo_envio > 5:
                payload = {
                    "morador_id": None,
                    "nome": "ALERTA_FRAUDE",
                    "authorized": False,
                    "gate": "closed",
                    "distance": 0.0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                client.publish(MQTT_TOPIC_ENVIO, json.dumps(payload))
                ultimo_envio = agora

        except Exception as e:
            print("Erro processamento rosto:", e)

    cv2.imshow("Controle de Acesso - Condominio", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()