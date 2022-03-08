import numpy as np
import onnxruntime
from math import ceil
from PIL import Image
from utils import Raster, pre_process

# TODO: save as geotif
# try:
#     from osgeo import gdal
# except:
#     import gdal


COLOR_MAP = [
    [240, 189, 0],  # cultivated land
    [232, 140, 181],  # garden land
    [44, 134, 133],  # forest land
    [118, 220, 169],  # grassland
    [116, 99, 255],  # residential area
    [147, 98, 189],  # industrial and mining land
    [135, 211, 249],  # traffic land
    [102, 143, 251],  # water area
    [104, 120, 156]   # unused land
]


def _save_palette(label, save_path):
    bin_colormap = np.zeros((256, 3))
    for i in range(9):
        bin_colormap[i, :] = COLOR_MAP[i]
    bin_colormap = bin_colormap.astype(np.uint8)
    visualimg  = Image.fromarray(label, "P")
    palette = bin_colormap
    visualimg.putpalette(palette) 
    visualimg.save(save_path, format='PNG')


if __name__ == "__main__":
    img_path = "E:/MyAIDatabase/landclas/nj.tif"
    onnx_path = "GhostNet_x1_3_9c.onnx"
    save_path = "E:/MyAIDatabase/landclas/landuse.png"
    block_size = 128
    ort_sess = onnxruntime.InferenceSession(onnx_path)  # classes is 8
    raster = Raster(img_path, to_uint8=True)
    rows = ceil(raster.height / block_size)
    cols = ceil(raster.width / block_size)
    landuse = np.zeros((rows, cols), dtype="uint8")
    total_number = int(rows * cols)
    for r in range(rows):
        for c in range(cols):
            loc_start = (c * block_size, r * block_size)
            title = raster.getArray(loc_start, (block_size, block_size))
            title = pre_process(title)
            ort_inputs = {ort_sess.get_inputs()[0].name: title}
            ort_outs = ort_sess.run(None, ort_inputs)[0]
            landuse[r, c] = np.argmax(ort_outs, axis=-1)
            print("-- {:d}/{:d} --".format(int(r * cols + c + 1), total_number))
    _save_palette(landuse, save_path)