import os
import sys
import argparse
import progressbar
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage import measure


'''
    (Usage)
    python preprocess_image.py --directory (absolute path)\
            --preprocess bitwise_downsampling\
            --value 4
'''


widgets=[
    ' [', progressbar.FormatLabel('Preprocess: %(value)4d / %(max_value)4d'), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]


def bitwise_downsampling(root_dir, bit_depth):
    if not os.path.exists(os.path.join(root_dir, '../preprocessed_images_bit/')):
        os.makedirs(os.path.join(root_dir, '../preprocessed_images_bit/'))
    
    if bit_depth < 1:
        bit_depth = 1
    if bit_depth > 8:
        bit_depth = 8

    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            filename, fileformat = os.path.splitext(filename)
            image = Image.open(os.path.join(folder, filename + fileformat))
            result = Image.eval(image, lambda pixel : (pixel - pixel % (256 >> bit_depth)))
            result.save(os.path.join(root_dir, '../preprocessed_images_bit/', filename + ".jpg"))

            bar_idx = bar_idx + 1
            bar.update(bar_idx)

    bar.finish()


def contrast_enhancing(root_dir, factor):
    if not os.path.exists(os.path.join(root_dir, '../preprocessed_images_contrast/')):
        os.makedirs(os.path.join(root_dir, '../preprocessed_images_contrast/'))

    if factor < 1:
        factor = 1
    if factor > 10:
        factor = 10

    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            filename, fileformat = os.path.splitext(filename)
            image = Image.open(os.path.join(folder, filename + fileformat))
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(factor)
            result.save(os.path.join(root_dir, '../preprocessed_images_contrast/', filename + ".jpg"))

            bar_idx = bar_idx + 1
            bar.update(bar_idx)

    bar.finish()


def color_quantization(root_dir, factor):
    if not os.path.exists(os.path.join(root_dir, '../preprocessed_images_kmeans/')):
        os.makedirs(os.path.join(root_dir, '../preprocessed_images_kmeans/'))

    if factor < 2:
        factor = 2
    if factor > 10:
        factor = 10

    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            filename, fileformat = os.path.splitext(filename)
            image = cv2.imread(os.path.join(folder, filename + fileformat))
            Z = image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(Z, factor, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res = res.reshape((image.shape))
            cv2.imwrite(os.path.join(root_dir, '../preprocessed_images_kmeans/', filename + ".jpg"), res)

            bar_idx = bar_idx + 1
            bar.update(bar_idx)

    bar.finish()


def target_segmentation(root_dir, factor):
    if not os.path.exists(os.path.join(root_dir, '../preprocessed_images_seg/')):
        os.makedirs(os.path.join(root_dir, '../preprocessed_images_seg/'))

    if factor < 1:
        factor = 1
    if factor > 10:
        factor = 10

    max_idx = sum([len(files) for r, d, files in os.walk(root_dir)])
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_idx)
    bar_idx = 0
    bar.start()

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            filename, fileformat = os.path.splitext(filename)
            image = Image.open(os.path.join(folder, filename + fileformat))
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(factor)
            result.save(os.path.join(root_dir, '../preprocessed_images_seg/', filename + ".jpg"))
            image = cv2.imread(os.path.join(root_dir, '../preprocessed_images_seg/', filename + ".jpg"), 0)
            row_size = image.shape[0]
            col_size = image.shape[1]
            thresh_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 2)
            labels = measure.label(thresh_image)
            regions = measure.regionprops(labels)
            selected_label = []
            for region in regions:
                bbox = region.bbox
                if bbox[2] - bbox[0] > row_size/15 or bbox[3] - bbox[1] > col_size/15:
                    selected_label.append(region.label)
            target = np.ndarray([row_size, col_size], dtype=np.int8)
            target[:] = 0
            for label in selected_label:
                target = target + np.where(labels == label, 255, 0)
            cv2.imwrite(os.path.join(root_dir, '../preprocessed_images_seg/', filename + ".jpg"), target)

            bar_idx = bar_idx + 1
            bar.update(bar_idx)

    bar.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="directory to preprocess")
    parser.add_argument("-p", "--preprocess", type=str, help="preprocessing method")
    parser.add_argument("-v", "--value", type=int, help="value for preprocessing", default=4)
    args = parser.parse_args()
    if(args.directory is None or args.preprocess is None or args.value is None):
        parser.print_help()
    else:
        if args.preprocess == "bitwise_downsampling":
            bitwise_downsampling(args.directory, args.value)
        elif args.preprocess == "contrast_enhancing":
            contrast_enhancing(args.directory, args.value)
        elif args.preprocess == "color_quantization":
            color_quantization(args.directory, args.value)
        elif args.preprocess == "target_segmentation":
            target_segmentation(args.directory, args.value)


if __name__ == "__main__":
    main()