import numpy as np
import cv2


def pre_process(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = (img / 255.0).astype("float32")
    img = img.transpose((2, 0, 1))[np.newaxis, :]
    return img