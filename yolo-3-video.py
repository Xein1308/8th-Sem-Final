import cv2
import numpy as np
import time

# Load video
video = cv2.VideoCapture('videos/traffic-cars.mp4')

# Load COCO labels
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# Load YOLO model
network = cv2.dnn.readNetFromDarknet(
    'yolo-coco-data/yolov3.cfg',
    'yolo-coco-data/yolov3.weights'
)

# Get output layers
layers_names_all = network.getLayerNames()

layers_names_output = [
    layers_names_all[i - 1]
    for i in network.getUnconnectedOutLayers().flatten()
]

# Detection parameters
probability_minimum = 0.5
threshold = 0.3

# Random colors
colors = np.random.randint(
    0,
    255,
    size=(len(labels), 3),
    dtype='uint8'
)

writer = None
h, w = None, None

frame_count = 0
total_time = 0

# Accuracy variables
total_confidence = 0
total_detections = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    if w is None or h is None:
        h, w = frame.shape[:2]

    # Create blob
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    # Forward pass
    network.setInput(blob)

    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    frame_count += 1
    total_time += (end - start)

    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Object Detection
    for result in output_from_network:

        for detected_objects in result:

            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:

                total_confidence += confidence_current
                total_detections += 1

                box_current = detected_objects[0:4] * np.array(
                    [w, h, w, h]
                )

                x_center, y_center, box_width, box_height = box_current

                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([
                    x_min,
                    y_min,
                    int(box_width),
                    int(box_height)
                ])

                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # Non-Maximum Suppression
    results = cv2.dnn.NMSBoxes(
        bounding_boxes,
        confidences,
        probability_minimum,
        threshold
    )

    # Draw Bounding Boxes
    if len(results) > 0:

        for i in results.flatten():

            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            color = colors[class_numbers[i]].tolist()

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_min + box_width, y_min + box_height),
                color,
                2
            )

            text = f'{labels[class_numbers[i]]}: {confidences[i]:.2f}'

            cv2.putText(
                frame,
                text,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # Calculate Accuracy
    if total_detections > 0:
        accuracy = (total_confidence / total_detections) * 100
    else:
        accuracy = 0

    # Calculate FPS
    fps = frame_count / total_time if total_time > 0 else 0

    # Display Accuracy
    cv2.putText(
        frame,
        f'Accuracy: {accuracy:.2f}%',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display FPS
    cv2.putText(
        frame,
        f'FPS: {fps:.1f}',
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Show Video
    cv2.imshow('YOLO Object Detection', frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Save Video
    if writer is None:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(
            'videos/result-traffic-cars.mp4',
            fourcc,
            30,
            (frame.shape[1], frame.shape[0]),
            True
        )

    writer.write(frame)

# Final Statistics
print("\n----- Final Results -----")
print("Total Frames:", frame_count)
print("Total Time:", round(total_time, 2), "seconds")

if total_time > 0:
    print("FPS:", round(frame_count / total_time, 1))

if total_detections > 0:
    final_accuracy = (total_confidence / total_detections) * 100
else:
    final_accuracy = 0

print("Average Detection Accuracy:",
      round(final_accuracy, 2), "%")

# Release Resources
video.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()