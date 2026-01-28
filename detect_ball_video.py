from ultralytics import YOLO
import os

def detect_tennis_ball_video(video_path, model_path='best.pt'):
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Load the model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Processing video: {video_path}...")
    # Run inference
    # save=True will save the annotated video to runs/detect/predict/
    results = model.predict(source=video_path, save=True, conf=0.25)

    print(f"\nResults saved to {results[0].save_dir}")

if __name__ == "__main__":
    video_path = "dataset_vid/v-zverev-jarry-rally-1.mp4"
    detect_tennis_ball_video(video_path)
