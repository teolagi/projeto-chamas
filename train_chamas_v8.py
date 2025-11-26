from ultralytics import YOLO
import torch # Adicionado para verificar a GPU

def main():
    # ...
    model = YOLO("yolov8n.pt") 

    model.train(
        data="chamas.yaml",
        epochs=200,
        device='0',  
        imgsz=640,
        batch=4,
        cache=True,
    )

if __name__ == '__main__':
    # Esta linha chama o código
    main()