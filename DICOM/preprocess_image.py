import numpy as np
import cv2


# summary of TODO
'''
    input: .png files (~/images/*)

    preprocessing methods (provided by argument)
    1. bitwise downsampling (LSB modification)
        for specific bit-depth, provided by argument
    2. contrast modification
    3. color quantization (Clustering)
        visit github (/asselinpaul/ImageSeg-KMeans) or
        http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#color-quantization
        (latter one is strongly recommended!)
    4. target segmentation
        Widely known as lung extraction, this method can be implemented
        with a few lines of openCV codes, please refer to
        https://www.raddq.com/jupyter/ProcessingDICOMInPython.html#Segmentation

    output .jpg files (~/images_(method_name)/*)

'''
