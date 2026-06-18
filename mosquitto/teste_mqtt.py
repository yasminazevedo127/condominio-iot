import paho.mqtt.client as mqtt

client = mqtt.Client()

client.username_pw_set(
    "mqtt",
    "mqtt123"
)

print("Conectando...")

client.connect(
    "192.168.64.2",
    1883,
    60
)

print("Conectado!")

client.publish(
    "teste",
    "Mensagem do Python"
)

print("Mensagem enviada!")

client.disconnect()