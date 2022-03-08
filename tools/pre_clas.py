import sys
import os.path as osp
sys.path.insert(0, osp.abspath("."))  # add workspace

import os
import argparse
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


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--data_folder", type=str, required=True, \
                    help="The folder of image titles data.")
parser.add_argument("--num_classes", type=int, default=8, \
                    help="The number of classes, `8` is the default.")
parser.add_argument("--save_folder", type=str, default="output", \
                    help="The folder path to save the results, `output` is the default.")
parser.add_argument("--onnx_path", type=str, default="GhostNet_x1_3.onnx", \
                    help="The path of onnx file, `GhostNet_x1_3.onnx` is the default.")


if __name__ == "__main__":
    # init
    args = parser.parse_args()
    data_folder = args.data_folder
    if not osp.exists(data_folder):
        raise ValueError("The `data_folder` is not exists!")
    num_classes = args.num_classes
    save_folder = args.save_folder
    onnx_path = args.onnx_path
    files = []
    features = []
    # mkdir
    _mkdirs(save_folder, num_classes)
    # extract feature
    names = os.listdir(data_folder)
    ort_sess = onnxruntime.InferenceSession(onnx_path)
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