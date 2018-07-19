# segnet_arc
ChainerCV SegNet for MC2ARCdataset

# Requirement
- Python 3.x / 2.x (Recommended version >= 2.7.11 or >= 3.5.1)
- OpenCV 3.x / 2.x
- Chainer 2.x (Recommended version == 2.1.0)
- ChainerCV 0.7.0
- Pillow(PIL) (Recommended version >= 3.1.0)
- numpy (Recommended version >= 1.10)
- matplotlib (Recommended version >= 1.4)

# Usage

## 1. Download the dataset
Please download the full-dataset(png) from the following URL.

- Web: http://mprg.jp/research/arc_dataset_2017_j
- Direct URL: http://www.mprg.cs.chubu.ac.jp/ARC2017/ARCdataset_png.zip

After the download completes, unzip "ARCdataset_png.zip".


## 2. Prepare the dataset for semantic segmentation
You need preparing the dataset for SegNet.
It is saved to new directory.
please change `make_data.py` as below.

```
# Original dataset dir (Source)
original_dataset = "<Your dataset path>"

# Dataset dir for SegNet (Destination)
segnet_dataset = "<New dataset path>"
```
`original_dataset` is path of your dataset which was downloaded.
`segnet_dataset` is new dataset path. For example, `<Your dataset path>/for_segnet/` would be good.

After setting path, Please run `make_dataset.py`.
```
$ python make_dataset.py
```


## 3. Get class weight
Please change `readARCdataset.py` as below.

```
root = "<Your dataset path>"
```
This is same as `segnet_dataset` of `make_data.py`.

And please run `calc_weight.py`.
```
$ python calc_weight.py
```

You can get `class_weight.npy` which is necessary file for training.


## 4. Training
Please run `train.py`.
```
$ python train.py
```

You can use following argments.
- `--gpu` (default -1) : # of GPU. Negative value indicates CPU.
- `--batchsize` (default 8)
- `--class_weight` (default 'class_weight.npy')
- `--out` (default 'result') : Output path of training results and models.

Note that default setting is NOT using GPU.
If you want to use GPU, please run as below.
```
$ python train.py --gpu <GPU ID>
```


## 5. Testing
Please run `test.py`.
```
$ python test.py --model <Trained model path> --output <segmentation result path>
```

You can use following argments.
- `--gpu` (default -1) : # of GPU. Negative value indicates CPU.
- `--model` : training model path.
- `--input` ( default <dataset path>/test)
- `--output` : Output path of segmented images(results).

`--model` is path of training model which was saved in `result` (or `--out` of `train.py`).
It was named `model_iteration-XXXXX`.
