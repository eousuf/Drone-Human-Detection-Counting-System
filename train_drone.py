from ultralytics import YOLO

def main():
    # Load the YOLOv8 Small model. 
    # 's' is "Small" - it's fast but quite accurate for beginners.
    model = YOLO('runs/detect/drone_model_v1/weights/last.pt') 

    # Start training
    model.train(
        data='visdrone.yaml',    # Points to your config file
        epochs=20,               # How many times the model sees the whole dataset
        imgsz=640,               # Resize images to 640x640 for training
        batch=16,                # Number of images processed at once
        device='cpu',                # Use 0 for an NVIDIA GPU, or 'cpu'
        name='drone_model_v1',    # The name of the output folder
        resume = True
    )

if __name__ == '__main__':
    main()