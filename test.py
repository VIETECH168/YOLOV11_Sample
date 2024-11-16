import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model (e.g., YOLO11n)
model = YOLO("yolo11n.pt")  # Replace with the actual path to your model file

# Set the model to only detect 'person' (class ID for 'person' is typically 0 in COCO dataset)
model.classes = [0]  # 0 is usually the class ID for "person"

# Open the video file
cap = cv2.VideoCapture('football.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if there are no frames left

    # Run inference
    results = model.predict(source=frame, imgsz=640, conf=0.5)

    # Draw bounding boxes and labels on the frame
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
        confidence = result.conf[0]
        class_id = int(result.cls[0])

        # Draw bounding box and label if the detected class is 'person'
        if class_id == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLO Person Detection", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
