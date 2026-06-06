import numpy as np
import cv2
import time

# Load image
image_BGR = cv2.imread('images/woman-working-in-the-office.jpg')

# Get image dimensions
h, w = image_BGR.shape[:2]

print('Image Shape:', image_BGR.shape)
print('Image Height =', h)
print('Image Width =', w)

# Create blob
blob = cv2.dnn.blobFromImage(
    image_BGR,
    1 / 255.0,
    (416, 416),
    swapRB=True,
    crop=False
)

print('Blob Shape:', blob.shape)

# Convert blob to image for display
blob_to_show = blob[0].transpose(1, 2, 0)
blob_to_show = cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR)

# Load labels
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# Load YOLO network
network = cv2.dnn.readNetFromDarknet(
    'yolo-coco-data/yolov3.cfg',
    'yolo-coco-data/yolov3.weights'
)

# Output layers
layers_names_all = network.getLayerNames()

layers_names_output = [
    layers_names_all[i - 1]
    for i in network.getUnconnectedOutLayers().flatten()
]

# Parameters
probability_minimum = 0.5
threshold = 0.3

# Colors
colors = np.random.randint(
    0,
    255,
    size=(len(labels), 3),
    dtype='uint8'
)

# Forward pass
network.setInput(blob)

start = time.time()

output_from_network = network.forward(layers_names_output)

end = time.time()

print('Detection Time:', round(end - start, 5), 'seconds')

# Lists
bounding_boxes = []
confidences = []
class_numbers = []

# Detection
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

# Draw bounding boxes
if len(results) > 0:

    for i in results.flatten():

        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]

        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        color = colors[class_numbers[i]].tolist()

        cv2.rectangle(
            image_BGR,
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
            image_BGR,
            text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

# SHOW ALL WINDOWS TOGETHER

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Blob Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Final Detection', cv2.WINDOW_NORMAL)

cv2.imshow('Original Image', cv2.imread('images/woman-working-in-the-office.jpg'))
cv2.imshow('Blob Image', blob_to_show)
cv2.imshow('Final Detection', image_BGR)

# Press any key to close all windows
cv2.waitKey(0)

cv2.destroyAllWindows()
