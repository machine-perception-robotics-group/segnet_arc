import argparse

import chainer
import numpy as np

from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from readARCdataset import arc_label_names
from readARCdataset import arcDataset

from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links import SegNetBasic

def transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--class_weight', type=str, default='class_weight.npy')
    parser.add_argument('--out', type=str, default='result')
    args = parser.parse_args()

    # # of classes
    classes = len(arc_label_names)

    # Triggers
    log_trigger = (50, 'iteration')
    validation_trigger = (2000, 'iteration')
    end_trigger = (100000, 'iteration')
    model_save_trigger = (10000, 'iteration')

    # Dataset
    train = arcDataset(split='train')
    train = TransformDataset(train, transform)
    val = arcDataset(split='val')

    # Iterator
    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    val_iter = iterators.MultiprocessIterator(
        val, args.batchsize, shuffle=False, repeat=False)

    # Model
    class_weight = np.load(args.class_weight)
    model = SegNetBasic(n_class = classes)
    model = PixelwiseSoftmaxClassifier(
        model, class_weight = class_weight)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    else:
        print("[Info] Trainer will use CPU.")

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=args.out)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss'], x_key='iteration', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['validation/main/miou'], x_key='iteration', file_name='miou.png'))

    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_iteration-{.updater.iteration}'),
        trigger=model_save_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr',
         'main/loss', 'validation/main/miou',
         'validation/main/mean_class_accuracy',
         'validation/main/pixel_accuracy']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        SemanticSegmentationEvaluator(
            val_iter, model.predictor,
            arc_label_names),
        trigger=validation_trigger)

    trainer.run()

if __name__ == '__main__':
    main()
