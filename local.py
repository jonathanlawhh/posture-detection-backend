import os
import main
import cv2
import base64
import numpy as np


def readb64(uri):
    nparr = np.frombuffer(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


if __name__ == "__main__":
    for f in os.listdir(os.path.join(os.getcwd(), "input")):
        file = {'file': open(os.path.join(os.getcwd(), "input", f), 'rb')}
        results = main.predict(file["file"])

        print(f, "[INFO] YOLO took", results[0]["duration"])
        print(f, results[0]["prediction"])
        print("")
        cv2.imshow(f, readb64(results[0]["image"]))

    cv2.waitKey(0)
