import glob
import os
import shutil
import cv2 as cv
import numpy as np

from distutils.dir_util import copy_tree

# Original dataset dir (Source)
original_dataset = "/Volumes/External/arcdataset/public/ARCdataset_png/"

# Dataset dir for SegNet (Destination)
segnet_dataset = "/Users/ryorsk/Desktop/ARCdataset_segnet_new/"

# Image extension (e.g. png, bmp)
ext = 'png'


class_color = np.array([
           [  0,   0,   0],
           [ 85,   0,   0],
           [170,   0,   0],
           [255,   0,   0],
           [  0,  85,   0],
           [ 85,  85,   0],
           [170,  85,   0],
           [255,  85,   0],
           [  0, 170,   0],
           [ 85, 170,   0],
           [170, 170,   0],
           [255, 170,   0],
           [  0, 255,   0],
           [ 85, 255,   0],
           [170, 255,   0],
           [255, 255,   0],
           [  0,   0,  85],
           [ 85,   0,  85],
           [170,   0,  85],
           [255,   0,  85],
           [  0,  85,  85],
           [ 85,  85,  85],
           [170,  85,  85],
           [255,  85,  85],
           [  0, 170,  85],
           [ 85, 170,  85],
           [170, 170,  85],
           [255, 170,  85],
           [  0, 255,  85],
           [ 85, 255,  85],
           [170, 255,  85],
           [255, 255,  85],
           [  0,   0, 170],
           [ 85,   0, 170],
           [170,   0, 170],
           [255,   0, 170],
           [  0,  85, 170],
           [ 85,  85, 170],
           [170,  85, 170],
           [255,  85, 170],
           [  0, 170, 170]])

class_color = class_color[:, ::-1]


def cvResize(src, w, h, inter):
    img = cv.imread(src)
    if img is None :
        print( "[ERROR]Cannot read image: " + src )
        sys.exit()
    img = cv.resize(img, (w, h), interpolation=inter)
    return img


def convertGray(img):
    for i in range(0, class_color.shape[0]):
        img[(img == class_color[i]).all(axis=2)] = [i, i, i]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def copyResize(src, dest, type):
    src_list = glob.glob(os.path.join(src, '*.' + ext))

    for src_path in src_list:
        print(src_path)
        if type == "color":
            img = cvResize(src_path, 480, 360, cv.INTER_CUBIC)
        elif type == "gray":
            img = cvResize(src_path, 480, 360, cv.INTER_NEAREST)
            img = convertGray(img)
        cv.imwrite(os.path.join(dest, os.path.basename(src_path)), img)


def saveFileList(image_dir, annot_dir, type):
    image_list = glob.glob(os.path.join(image_dir, '*.' + ext))
    image_list.sort()

    annot_list = glob.glob(os.path.join(annot_dir, '*.' + ext))
    annot_list.sort()

    if len(image_list) != len(annot_list):
        print("[Error] Number of files is mismatch: " + image_dir + ", " + annot_dir )
        exit(1)

    f = open(os.path.join(segnet_dataset, type + ".txt"), mode='w')
    for i in range(0, len(image_list)):
        f.write(os.path.join(type, os.path.basename(image_list[i])))
        f.write(' ')
        f.write(os.path.join(type + "annot", os.path.basename(annot_list[i])))
        f.write('\n')
    f.close()


def moveValidationSet(train, trainannot, val, valannot):
    image_list = glob.glob(os.path.join(train, '*.' + ext))
    image_list.sort()

    annot_list = glob.glob(os.path.join(trainannot, '*.' + ext))
    annot_list.sort()

    if len(image_list) != len(annot_list):
        print("[Error] Number of files is mismatch: " + image_dir + ", " + annot_dir )
        exit(1)

    for i in range(0, len(image_list)):
        if len(os.path.basename(image_list[i])) >= 15: break
        if (i % 16) <= 3:
            #print(image_list[i])
            shutil.move(image_list[i], val)
            shutil.move(annot_list[i], valannot)


def main():
    # Source dir
    src_test_path       = os.path.join(original_dataset, "test_known", "rgb")
    src_testannot_path  = os.path.join(original_dataset, "test_known", "segmentation")
    src_train_path      = os.path.join(original_dataset, "train", "rgb")
    src_trainannot_path = os.path.join(original_dataset, "train", "segmentation")

    # Destination dir init
    dest_test_path       = os.path.join(segnet_dataset, "test")
    dest_testannot_path  = os.path.join(segnet_dataset, "testannot")
    dest_train_path      = os.path.join(segnet_dataset, "train")
    dest_trainannot_path = os.path.join(segnet_dataset, "trainannot")
    dest_val_path        = os.path.join(segnet_dataset, "val")
    dest_valannot_path   = os.path.join(segnet_dataset, "valannot")
    [os.mkdir(path) for path in [dest_test_path, dest_testannot_path,
        dest_train_path, dest_trainannot_path, dest_val_path, dest_valannot_path]]

    # Copy images
    copyResize(src_test_path, dest_test_path, "color")
    copyResize(src_testannot_path, dest_testannot_path, "gray")
    copyResize(src_train_path, dest_train_path, "color")
    copyResize(src_trainannot_path, dest_trainannot_path, "gray")

    # Move to val
    moveValidationSet(dest_train_path, dest_trainannot_path, dest_val_path, dest_valannot_path)

    # Text export
    saveFileList(dest_test_path, dest_testannot_path, "test")
    saveFileList(dest_train_path, dest_trainannot_path, "train")
    saveFileList(dest_val_path, dest_valannot_path, "val")

    print("[Info] Done.")

if __name__ == '__main__':
    main()
