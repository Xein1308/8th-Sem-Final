import cv2
import numpy as np
import time

# Open webcam
camera = cv2.VideoCapture(0)

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

# Parameters
probability_minimum = 0.5
threshold = 0.3

# Random colors
colors = np.random.randint(
    0,
    255,
    size=(len(labels), 3),
    dtype='uint8'
)

h, w = None, None

while True:

    ret, frame = camera.read()

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

    print('Frame Time:', round(end - start, 5), 'seconds')

    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Detection loop
    for result in output_from_network:

        for detected_objects in result:

            scores = detected_objects[5:]

            class_current = np.argmax(scores)

            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:

                box_current = detected_objects[0:4] * np.array([w, h, w, h])

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

    # Draw boxes
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

            text = '{}: {:.2f}'.format(
                labels[class_numbers[i]],
                confidences[i]
            )

            cv2.putText(
                frame,
                text,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Show webcam output
    cv2.namedWindow(
        'YOLO Real-Time Detection',
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        'YOLO Real-Time Detection',
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()

cv2.destroyAllWindows()
