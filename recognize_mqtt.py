import cv2
import face_recognition
import paho.mqtt.client as mqtt
import json
import time

# -----------------------
# MQTT
# -----------------------

client = mqtt.Client()

client.connect(
    "192.168.64.2",
    1883,
    60
)

# -----------------------
# Carrega rosto conhecido
# -----------------------

yasmin_image = face_recognition.load_image_file(
    "known_faces/yasmin.jpg"
)

yasmin_encoding = face_recognition.face_encodings(
    yasmin_image
)[0]

known_encodings = [
    yasmin_encoding
]

known_names = [
    "Yasmin"
]

# -----------------------
# Webcam
# -----------------------

video = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

ultimo_envio = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    face_locations = face_recognition.face_locations(
        rgb
    )

    face_encodings = face_recognition.face_encodings(
        rgb,
        face_locations
    )

    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding
        )

        name = "Desconhecido"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        top, right, bottom, left = face_location

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        agora = time.time()

        if name != "Desconhecido":

            if agora - ultimo_envio > 5:

                payload = {
                    "name": name,
                    "authorized": True
                }

                client.publish(
                    "face/recognition",
                    json.dumps(payload)
                )

                print(
                    f"Reconhecido: {name}"
                )

                ultimo_envio = agora

    cv2.imshow(
        "Reconhecimento Facial",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
