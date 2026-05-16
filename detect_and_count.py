import os
import cv2
from ultralytics import YOLO

# File to store the last viewed image index so we can resume later
PROGRESS_FILE = 'viewer_progress.txt'

def load_last_index():
    """Loads the last viewed image index from a file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return int(f.read().strip())
        except ValueError:
            return 0
    return 0

def save_current_index(index):
    """Saves the current image index to a file."""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))

def main():
    # Load your trained model weights
    model = YOLO('runs/detect/drone_model_v1/weights/best.pt')

    # Automatically find the absolute folder path where THIS script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(BASE_DIR, 'VisDrone_Dataset', 'VisDrone2019-DET-test-dev', 'images')
    
    if not os.path.exists(images_dir):
        print(f"Error: The directory '{images_dir}' could not be found.")
        return

    # List and sort all image files to keep the index order consistent
    all_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not all_images:
        print(f"Error: No image files found inside '{images_dir}'!")
        return
        
    total_images = len(all_images)
    
    current_index = load_last_index()
    if current_index >= total_images or current_index < 0:
        current_index = 0

    print(f"Found {total_images} images. Resuming from image index: {current_index}")
    print("--> Press 'd' key for Next image.")
    print("--> Press 'a' key for Previous image.")
    print("--> Press 'ESC' or click the window 'X' close button to Exit.")
    print("-" * 50)

    # Main interactive loop
    while True:
        image_name = all_images[current_index]
        image_path = os.path.join(images_dir, image_name)
        
        img = cv2.imread(image_path)
        if img is None:
            # Skip broken images safely
            current_index = (current_index + 1) % total_images
            continue

        results = model.predict(source=image_path, conf=0.15, verbose=False)[0]

        human_count = 0

        # Draw bounding boxes
        for box in results.boxes:
            class_id = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            if class_id == 0:  # Pedestrian
                human_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, 'Human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif class_id == 3:  # Car
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw Overlay Text Info Block
        count_text = f"[{current_index + 1}/{total_images}] File: {image_name} | Humans: {human_count}"
        cv2.rectangle(img, (10, 10), (550, 50), (0, 0, 0), -1)
        cv2.putText(img, count_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        window_name = 'Drone Detection & Counting Pipeline'
        cv2.imshow(window_name, img)
        
        # waitKey(0) captures basic keys;getWindowProperty checks if window 'X' was pressed
        key = cv2.waitKey(0) & 0xFF
        
        # Check if the user clicked the window's 'X' exit button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed via GUI close button.")
            save_current_index(current_index)
            break

        # 27 is the Escape key
        if key == 27:
            print("Escape key pressed. Exiting and saving progress...")
            save_current_index(current_index)
            break
            
        # Press 'a' to go PREVIOUS (Left)
        elif key == ord('a'):  
            if current_index > 0:
                current_index -= 1
            else:
                current_index = total_images - 1  # Loop back to end
                
        # Press 'd' or 'Spacebar' to go NEXT (Right)
        elif key == ord('d') or key == 32:  
            current_index = (current_index + 1) % total_images

    cv2.destroyAllWindows()
    print("Pipeline closed cleanly.")

if __name__ == '__main__':
    main()