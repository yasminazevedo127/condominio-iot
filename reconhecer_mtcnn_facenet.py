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

# =========================
# CONFIGURAÇÕES
# =========================

LIMIAR_RECONHECIMENTO = 0.9
LIMIAR_ANTI_SPOOFING = 0.6  # Ajuste entre 0.5 e 0.7 dependendo da iluminação

MQTT_HOST = "192.168.64.2"
MQTT_PORT = 1883
MQTT_TOPIC_ENVIO = "condominio/acesso"
MQTT_TOPIC_STATUS = "condominio/portao/status"

portao_status = "idle"
cooldowns_moradores = {}
ultimo_envio = 0

# =========================
# CALLBACKS MQTT
# =========================

def on_message(client, userdata, msg):
    global portao_status
    if msg.topic == MQTT_TOPIC_STATUS:
        portao_status = msg.payload.decode().strip()
        print(f"[MQTT] Status do portão atualizado para: {portao_status}")

# =========================
# CONFIGURAÇÃO MQTT
# =========================

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set("mqtt", "mqtt123")
client.on_message = on_message

print("Conectando ao MQTT...")
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_STATUS)
client.loop_start()

# =========================
# CARREGAMENTO DOS MODELOS
# =========================

detector = MTCNN()
embedder = FaceNet()

# Inicializa a sessão com o novo modelo MiniFASNetV2 do repositório
caminho_onnx = "modelos/MiniFASNetV2.onnx"
if not os.path.exists(caminho_onnx):
    raise FileNotFoundError(f"Coloque o arquivo baixado em: {caminho_onnx}")

ort_session = ort.InferenceSession(caminho_onnx, providers=['CPUExecutionProvider'])
nome_entrada = ort_session.get_inputs()[0].name

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
# FUNÇÃO AUXILIAR ANTI-SPOOFING
# =========================

def verificar_vivacidade(rosto_img):
    """
    Processa o recorte do rosto usando a arquitetura MiniFASNetV2 via ONNX.
    """
    try:
        # Redimensiona para 80x80 (padrão do MiniFASNetV2)
        rosto_redimensionado = cv2.resize(rosto_img, (80, 80))
        
        # Pré-processamento: converte para float32 e normaliza os dados
        img_processada = rosto_redimensionado.astype(np.float32)
        img_processada = np.transpose(img_processada, (2, 0, 1))  # Altera de HWC para CHW
        img_processada = np.expand_dims(img_processada, axis=0)     # Cria o batch (1, 3, 80, 80)
        
        # Roda a inferência no ONNX Runtime
        saidas = ort_session.run(None, {nome_entrada: img_processada})
        saida = saidas[0]
        
        # Softmax para extrair as probabilidades
        exp_saida = np.exp(saida - np.max(saida))
        probabilidades = exp_saida / exp_saida.sum()
        
        # O modelo retorna as classes onde o índice 1 é a pessoa viva (Real)
        probabilidade_real = probabilidades[0][1]
        return probabilidade_real
    except Exception as e:
        print("Erro ao executar Anti-Spoofing:", e)
        return 0.0

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("Sistema Iniciado com Anti-Spoofing Ativo. Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for face in faces:
        try:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)
            rosto = rgb[y:y+h, x:x+w]

            if rosto.size == 0:
                continue

            # Faz a predição se o rosto diante da lente é real ou fraude
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
                menor_distancia = 0.0
                print(f"[BLOQUEADO] Tentativa de fraude! Score Real: {score_real:.2f}")

            # ---------------------------
            # Interface Visual Dinâmica
            # ---------------------------
            if not eh_real:
                cor = (0, 0, 255) # Vermelho
                status_txt = f"FOTO DETECTADA ({score_real:.2f})"
            elif autorizado:
                if portao_status != "idle":
                    cor = (255, 165, 0) # Laranja
                    status_txt = "Portao Ocupado"
                elif melhor_id in cooldowns_moradores and (agora - cooldowns_moradores[melhor_id] <= 30):
                    cor = (0, 255, 0)
                    status_txt = "Acesso Liberado"
                else:
                    cor = (0, 255, 0) # Verde
                    status_txt = f"Real: {nome}"
            else:
                cor = (0, 0, 255)
                status_txt = "Desconhecido"

            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(frame, status_txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            # ---------------------------
            # Lógica de Envio MQTT
            # ---------------------------
            if autorizado and portao_status == "idle" and eh_real:
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
                    registrar_acesso(morador_id=melhor_id, autorizado=1, distancia=float(menor_distancia), data_hora=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                    print(f"[SUCESSO] Comando de abertura enviado para {nome}.")
                    cooldowns_moradores[melhor_id] = agora
            
            elif not eh_real and agora - ultimo_envio > 5:
                payload = {"morador_id": None, "nome": "ALERTA_FRAUDE", "authorized": False, "gate": "closed", "distance": 0.0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                client.publish(MQTT_TOPIC_ENVIO, json.dumps(payload))
                ultimo_envio = agora

        except Exception as e:
            print("Erro interno de processamento:", e)

    cv2.imshow("Controle de Acesso - Condominio", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
