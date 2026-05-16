import os
import cv2
from ultralytics import YOLO

def main():
    model = YOLO('runs/detect/drone_model_v1/weights/best.pt')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(BASE_DIR, 'VisDrone_Dataset', 'VisDrone2019-DET-test-dev', 'images')
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory '{images_dir}' not found.")
        return

    all_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not all_images:
        print("No images found to sequence.")
        return

    print(f"Sequencing {len(all_images)} images into a simulated video feed...")
    print("Press 'q' inside the playback window to stop tracking.")
    print("-" * 50)

    #Process the sequenced images
    for image_name in all_images:
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        #Run YOLOv8 Tracking using ByteTrack
        # 'persist=True' tells the model to remember objects from the previous frame
        results = model.track(source=image_path, persist=True, tracker="bytetrack.yaml", conf=0.15, verbose=False)[0]

        # Initialize tracking counters for this frame
        frame_human_count = 0
        frame_car_count = 0

        # Check if any objects were successfully tracked in this frame
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            # Zip them together to loop through coordinates, class, and unique ID
            for box, cls, track_id in zip(boxes, classes, track_ids):
                x1, y1, x2, y2 = box

                # Handle Pedestrians (Class 0)
                if cls == 0:
                    frame_human_count += 1
                    # Draw Green box and include the unique Tracking ID
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Human ID: {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Handle Cars (Class 3)
                elif cls == 3:
                    frame_car_count += 1
                    # Draw Blue box and include the unique Tracking ID
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"Car ID: {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 5. Add an information banner to the frame
        info_text = f"Frame Humans: {frame_human_count} | Frame Cars: {frame_car_count}"
        cv2.rectangle(img, (10, 10), (450, 50), (0, 0, 0), -1)
        cv2.putText(img, info_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 6. Stream the playback window
        cv2.imshow('Drone Tracking Pipeline (ByteTrack)', img)
        
        # We use a small delay (100ms) to create a video-like playback effect automatically
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("Tracking manual stop triggered.")
            break

    cv2.destroyAllWindows()
    print("Tracking pipeline finished.")

if __name__ == '__main__':
    main()