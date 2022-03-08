import sys
import os.path as osp
sys.path.insert(0, osp.abspath("."))  # add workspace

import os
import shutil
import numpy as np
import onnxruntime
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils import pre_process


def _mkdirs(save_folder, number):
    for i in range(number):
        path = osp.join(save_folder, str(i))
        if not osp.exists(path):
            os.makedirs(path)
    print("Create successfully!")


if __name__ == "__main__":
    # init
    data_folder = "dataset"
    save_folder = "output"
    num_classes = 8
    files = []
    features = []
    # mkdir
    _mkdirs(save_folder, num_classes)
    # extract feature
    names = os.listdir(data_folder)
    ort_sess = onnxruntime.InferenceSession("GhostNet_x1_3.onnx")
    for name in tqdm(names):
        img_path = osp.join(data_folder, name)
        x = pre_process(img_path)
        ort_inputs = {ort_sess.get_inputs()[0].name: x}
        ort_outs = ort_sess.run(None, ort_inputs)[0][0]
        files.append(name)
        features.append(ort_outs)
    features = np.array(features)
    print(features.shape)
    print(features.shape)
    # kmeans
    kmeans = KMeans(n_clusters=num_classes) 
    kmeans.fit(features)
    print("Kmeans successfully!")
    # move file
    labels = kmeans.labels_
    for f, lab in tqdm(zip(files, labels)):
        raw_path = osp.join(data_folder, f)
        save_path = osp.join(save_folder, str(lab), f)
        shutil.copy(raw_path, save_path)