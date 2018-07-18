import os
import numpy as np

import chainer
from chainercv.utils import read_image

# Dataset dir
root = "/home/ryorsk/ARCdataset_segnet_new/"

arc_label_names = (
    "BG",
    "Avery_Binder",
    "Balloons",
    "Baby_Wipes",
    "Toilet_Brush",
    "Colgate_Toothbrush",
    "Crayons",
    "Epsom_Salts",
    "Robots_DVD",
    "Glue_Sticks",
    "Expo_Eraser",
    "Fiskars_Scissors",
    "Composition_Book",
    "Hanes_Socks",
    "Irish_Spring_Soap",
    "Band_Aid_Tape",
    "Tissue_Box",
    "Black_Fashion_Gloves",
    "Laugh_Out_Loud_Jokes",
    "Mesh_Cup",
    "Marbles",
    "Hand_Weight",
    "Plastic_Wine_Glass",
    "Poland_Spring_Water",
    "Pie_Plates",
    "Reynolds_Wrap",
    "Robots_Everywhere",
    "Duct_Tape",
    "Scotch_Sponges",
    "Speed_Stick",
    "Index_Cards",
    "Ice_Cube_Tray",
    "Table_Cloth",
    "Measuring_Spoons",
    "Bath_Sponge",
    "Ticonderoga_Pencils",
    "Mouse_Traps",
    "White_Facecloth",
    "Tennis_Ball",
    "Windex",
    "Flashlight"
)

arc_label_colors = (
    (   0,    0,    0),
    (  85,    0,    0),
    ( 170,    0,    0),
    ( 255,    0,    0),
    (   0,   85,    0),
    (  85,   85,    0),
    ( 170,   85,    0),
    ( 255,   85,    0),
    (   0,  170,    0),
    (  85,  170,    0),
    ( 170,  170,    0),
    ( 255,  170,    0),
    (   0,  255,    0),
    (  85,  255,    0),
    ( 170,  255,    0),
    ( 255,  255,    0),
    (   0,    0,   85),
    (  85,    0,   85),
    ( 170,    0,   85),
    ( 255,    0,   85),
    (   0,   85,   85),
    (  85,   85,   85),
    ( 170,   85,   85),
    ( 255,   85,   85),
    (   0,  170,   85),
    (  85,  170,   85),
    ( 170,  170,   85),
    ( 255,  170,   85),
    (   0,  255,   85),
    (  85,  255,   85),
    ( 170,  255,   85),
    ( 255,  255,   85),
    (   0,    0,  170),
    (  85,    0,  170),
    ( 170,    0,  170),
    ( 255,    0,  170),
    (   0,   85,  170),
    (  85,   85,  170),
    ( 170,   85,  170),
    ( 255,   85,  170),
    (   0,  170,  170)
)

class arcDataset(chainer.dataset.DatasetMixin):
    def __init__(self, split='train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError(
                'Please pick split from \'train\', \'val\', \'test\'')

        data_dir = root

        img_list_path = os.path.join(data_dir, '{}.txt'.format(split))
        self.paths = [[data_dir for fn in line.split()] for line in open(img_list_path)]

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        if i >= len(self):
            raise IndexError('index is too large')
        img_path, label_path = self.paths[i]
        img = read_image(img_path, color=True)
        label = read_image(label_path, dtype=np.int32, color=False)[0]
        return img, label
