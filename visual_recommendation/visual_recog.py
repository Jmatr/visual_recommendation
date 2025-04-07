import cv2
import torch
import argparse


def process_video(video_path, frame_rate=1.0):
    """
    Processes the video file at `video_path` by sampling frames and running object detection.

    Args:
        video_path (str): Path to the video file.
        frame_rate (float): Number of frames to process per second.

    Returns:
        dict: Aggregated object detections with object names as keys and counts as values.
    """
    # Load the pre-trained YOLOv5 model (this may download the model on the first run)
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    # The model is set to evaluation mode by default

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return {}

    # Get frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Fallback if FPS is not available
    frame_interval = max(1, int(fps / frame_rate))
    print(f"Video FPS: {fps}, processing every {frame_interval} frame(s).")

    frame_count = 0
    detections = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame
        if frame_count % frame_interval == 0:
            # Run object detection on the current frame
            results = model(frame)
            # Get a DataFrame with the detections (requires pandas)
            df = results.pandas().xyxy[0]
            labels = df['name'].tolist()
            for label in labels:
                detections[label] = detections.get(label, 0) + 1

        frame_count += 1

    cap.release()
    return detections


def main():
    parser = argparse.ArgumentParser(description="Video Object Detection using YOLOv5")
    parser.add_argument("video_path", type=str, help="Path to the video file to process")
    parser.add_argument("--frame_rate", type=float, default=1.0,
                        help="Number of frames to process per second (default: 1)")
    args = parser.parse_args()

    detections = process_video(args.video_path, args.frame_rate)
    if detections:
        print("\nAggregated detections:")
        for label, count in detections.items():
            print(f"{label}: {count}")
    else:
        print("No detections found or an error occurred.")


# python visual_recog.py test.mp4 --frame_rate 1.0
if __name__ == "__main__":
    main()
