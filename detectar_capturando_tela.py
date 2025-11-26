from ultralytics import YOLO
import cv2
from windowcapture import WindowCapture
import serial 
import numpy as np
import time

# ------------------- CONFIGURAÇÃO SERIAL -------------------
try:
    ser = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)  # Aguarda Arduino inicializar
    print("Conexão Serial com Arduino OK.")
except Exception as e:
    print(f"Erro ao conectar ao Arduino: {e}")
    ser = None

# ------------------- CAPTURA DE JANELA -------------------
offset_x = 100
offset_y = 100
capture_w = 800
capture_h = 600
wincap = WindowCapture(size=(capture_w, capture_h), origin=(offset_x, offset_y))

# ------------------- MODELO -------------------
model = YOLO("best.pt")

# ------------------- VARIÁVEIS DE ESTADO -------------------
ultima_classe_detectada = None
LIMITE_CONFIANCA = 70.0  

# ------------------- LOOP PRINCIPAL -------------------
while True:
    img = wincap.get_screenshot()
    if img is None or img.size == 0:
        print("Nenhuma imagem capturada, esperando...")
        time.sleep(0.5)
        continue

    results = model(img, device='cpu')
    annotated = results[0].plot()
    cv2.imshow("Resultado IA", annotated)
    cv2.moveWindow("Resultado IA", 1200, 100)

    # ------------------- PROCESSAMENTO DA DETECÇÃO -------------------
    if len(results[0].boxes.cls) > 0 and ser:
        class_id = int(results[0].boxes.cls[0].item())
        nome_da_classe = results[0].names[class_id]
        conf = results[0].boxes.conf[0].item() * 100
        print(f"Detectado: {nome_da_classe} ({conf:.1f}%)")

        if conf >= LIMITE_CONFIANCA:
            comando = None
            if nome_da_classe == 'carburante':
                comando = '0\n'
            elif nome_da_classe == 'neutra':
                comando = '1\n'
            elif nome_da_classe == 'oxidante':
                comando = '2\n'

            # Só envia se for diferente do anterior
            if comando and nome_da_classe != ultima_classe_detectada:
                ser.write(comando.encode('utf-8'))
                ser.flush()
                ultima_classe_detectada = nome_da_classe
                print(f">>> Comando enviado: {comando.strip()}")
        else:
            print("Confiança abaixo do limite.")
    else:
        # Nenhuma chama detectada
        if ultima_classe_detectada is not None and ser:
            ser.write(b'9\n')
            ser.flush()
            print(">>> Nenhuma chama detectada. Comando '9' enviado.")
            ultima_classe_detectada = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- ENCERRAMENTO -------------------
if ser:
    ser.close()
cv2.destroyAllWindows()
print("Desligando...")