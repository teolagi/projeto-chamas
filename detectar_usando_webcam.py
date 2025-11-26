from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import serial
import time

# ---------------------------
# CONFIGURAÇÃO SERIAL
# ---------------------------
try:
    ser = serial.Serial('COM5', 9600, timeout=0.1)  # Ajuste a porta COM se necessário
    time.sleep(2)  # Aguarda o Arduino reiniciar após abrir a porta serial
    print("Conexão Serial com Arduino OK.")
except Exception as e:
    print(f"Erro ao conectar ao Arduino: {e}")
    ser = None

# ---------------------------
# CONFIGURAÇÃO DO MODELO E VÍDEO
# ---------------------------
cap = cv2.VideoCapture(0)  # Usa a webcam
model = YOLO("best.pt")    # Seu modelo YOLO treinado

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True
ultima_classe_detectada = None  # Armazena a última classe enviada para evitar repetição

# ---------------------------
# LOOP PRINCIPAL
# ---------------------------
while True:
    success, img = cap.read()
    if not success:
        break

    # Faz a detecção
    results = model.track(img, persist=True, device='cpu') if seguir else model(img, device='cpu')

    for result in results:
        if len(result.boxes.cls) > 0:
            # Pega a primeira detecção
            class_id = int(result.boxes.cls[0].item())
            nome_classe = result.names[class_id]
            confianca = result.boxes.conf[0].item() * 100

            print(f"Detectado: {nome_classe} com {confianca:.1f}% de confiança")

            # ENVIO PARA O ARDUINO (SE PASSAR DO LIMITE DE CONFIANÇA)
            LIMITE_CONFIANCA = 65.0
            if confianca >= LIMITE_CONFIANCA and ser:
                comando = None

                if nome_classe == 'carburante':
                    comando = '0\n'
                elif nome_classe == 'neutra':
                    comando = '1\n'
                elif nome_classe == 'oxidante':
                    comando = '2\n'

                # Só envia se for diferente do anterior
                if comando and nome_classe != ultima_classe_detectada:
                    ser.write(comando.encode('utf-8'))
                    ser.flush()
                    ultima_classe_detectada = nome_classe
                    print(f">>> Comando enviado: {comando.strip()}")
            else:
                print("Confiança baixa ou sem conexão serial.")
        else:
            # Nenhuma chama detectada: envia "reset" apenas se a última não for None
            if ultima_classe_detectada is not None and ser:
                ser.write('9\n'.encode('utf-8'))  # 9 = comando "nenhuma chama"
                ser.flush()
                print(">>> Nenhuma chama detectada. Comando '9' enviado.")
                ultima_classe_detectada = None

        # PLOTAGEM DOS RESULTADOS
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except:
                pass

    # MOSTRA O FRAME NA TELA
    cv2.imshow("Tela", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# ENCERRAMENTO
# ---------------------------
cap.release()
if ser:
    ser.close()
    print("Porta Serial fechada.")
cv2.destroyAllWindows()
print("Desligando...")