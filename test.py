import argparse
import glob
import os

import matplotlib
from matplotlib.patches import Patch
from PIL import Image
import numpy as np

import chainer
from chainercv.links import SegNetBasic
from chainercv import utils

from readARCdataset import arc_label_names
from readARCdataset import arc_label_colors
from readARCdataset import root


def vis_semantic_segmentation(
        label, label_names=None, label_colors=None, alpha=1):

    if label_names is not None:
        n_class = len(label_names)
    elif label_colors is not None:
        n_class = len(label_colors)
    else:
        n_class = label.max() + 1

    if label_colors is not None and not len(label_colors) == n_class:
        raise ValueError(
            'The size of label_colors is not same as the number of classes')
    if label.max() >= n_class:
        raise ValueError('The values of label exceed the number of classes')

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    if label_colors is None:
        print("[Error] label_color is None")
        exit(1)

    label_colors = np.array(label_colors) / 255.
    cmap = matplotlib.colors.ListedColormap(label_colors)

    img = cmap(label / float(n_class - 1), alpha=alpha)

    img = img * 255
    return(img)


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model')
    parser.add_argument('--input', default=os.path.join(root, "test"))
    parser.add_argument('--output')
    args = parser.parse_args()

    model = SegNetBasic(
        n_class=len(arc_label_names),
        pretrained_model=args.model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    image_list = glob.glob(args.input + "/*.png")
    image_list.sort()

    for image_path in image_list:
        print(image_path)

        img = utils.read_image(image_path, color=True)
        labels = model.predict([img])
        label = labels[0]

        segimg = vis_semantic_segmentation(label, arc_label_names, arc_label_colors)
        Image.fromarray(np.uint8(segimg)).save(args.output + image_path[image_path.rfind('/'):])


if __name__ == '__main__':
    main()
