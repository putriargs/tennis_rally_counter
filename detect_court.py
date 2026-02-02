import cv2
import os
from tennis_court_detector.tennis_court_detector import TennisCourtDetector

def detect_court_lines(image_path, model_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Initialize the detector
    detector = TennisCourtDetector(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Predict keypoints
    keypoints = detector.predict(image)

    # Draw keypoints
    output_image = detector.draw_keypoints(image.copy(), keypoints)

    # Save the output
    output_dir = "runs/detect_court"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, output_image)
    print(f"Court detection result saved to {output_path}")

    # Optional: Display the keypoints coordinates
    print(f"Keypoints detected for {filename}:")
    for i in range(0, len(keypoints), 2):
        print(f"Point {i//2}: ({keypoints[i]:.2f}, {keypoints[i + 1]:.2f})")

if __name__ == "__main__":
    image_path = "dataset_img/js-ball-match-3.png" # Example image
    model_path = "models/keypoints_model_50.pth"
    detect_court_lines(image_path, model_path)
