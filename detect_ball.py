from ultralytics import YOLO
import os

def detect_tennis_ball(image_path, model_path='best.pt'):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    # save=True will save the annotated image to runs/detect/predict/
    results = model.predict(source=image_path, save=True, conf=0.25)
    
    # Process results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} tennis ball(s).")
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            print(f"Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], Confidence: {conf:.2f}")

    print(f"\nResults saved to {results[0].save_dir}")

if __name__ == "__main__":
    image_path = "dataset_img/js-ball-match-3.png"
    detect_tennis_ball(image_path)
