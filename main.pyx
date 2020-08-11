import cv2
import time
import numpy as np
cimport numpy as np
import json
import base64

cdef int INPUT_SIZE = 256
cdef list NAMES = [str]

# Preload models for performance
print("[INFO] loading YOLO...")
net = cv2.dnn.readNetFromDarknet("./yolo_configs/yolov3-obj.cfg", "./yolo_configs/posture_yolov3.weights")

print("[INFO] loading labels...")
with open("./yolo_configs/posture.names", 'rt') as f:
    NAMES = f.read().rstrip('\n').split('\n')

# Assign colors for drawing bounding boxes
cdef list COLORS = [
    [0, 200, 0], [20, 45, 144],
    [157, 224, 173], [0, 0, 232],
    [26, 147, 111], [40, 44, 100]
]


cdef np.ndarray[unsigned char, ndim=3] rescale_image(np.ndarray[unsigned char, ndim=3] input_img):
    cdef int h = input_img.shape[0]
    cdef int w = input_img.shape[1]

    # Resize if height is more than 1000px. First numerical + 1, will be the ratio to scale to.
    # Eg. 2540px, 2540px / ( 2 + 1 ) = new height.
    return input_img if h < 1000 else cv2.resize(input_img, (int(w / (int(str(h)[0]) + 2)), int(h / (int(str(h)[0]) + 2))))


cdef list predict_yolo(np.ndarray[unsigned char, ndim=3] input_img):
    cdef list ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cdef np.ndarray blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)

    return net.forward(ln)


cdef list draw_bound(np.ndarray[unsigned char, ndim=3] input_img, list layer_outputs, float confidence_level, float threshold):
    cdef list boxes = []
    cdef list confidences = []
    cdef list class_id = []
    cdef list results = []
    cdef list color = []

    cdef int H = input_img.shape[0]
    cdef int W = input_img.shape[1]
    cdef int x, y, centerX, centerY, width, height

    cdef float confidence

    cdef str text

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_ids = np.argmax(scores)
            confidence = scores[class_ids]

            if confidence > confidence_level:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_id.append(class_ids)

    # Non maxima
    cdef np.ndarray idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[class_id[i]]]
            cv2.rectangle(input_img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(NAMES[class_id[i]], confidences[i])
            cv2.putText(input_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            results.append([NAMES[class_id[i]], confidences[i]])

    return [input_img, results]


cpdef list predict(f):
    cdef double start = time.time()
    cdef np.ndarray[unsigned char, ndim=3] im = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)
    im = rescale_image(im)

    cdef list layer_outputs = predict_yolo(im)
    cdef list results = draw_bound(im, layer_outputs, 0.5, 0.4)

    ret, buffer = cv2.imencode('.png', results[0])
    cdef str encoded_im = base64.b64encode(buffer).decode()
    cdef double end = time.time()

    return [{
        "prediction": json.dumps(results[1]),
        "image": encoded_im,
        "duration": "{:.4f} seconds".format(end - start)
    }]
