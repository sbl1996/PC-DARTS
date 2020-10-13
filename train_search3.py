import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from hinas.train.darts import DARTSLearner
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

from torch.optim import SGD, Adam

from horch.common import convert_tensor
from horch.datasets import train_test_split
from horch.defaults import set_defaults
from horch.optim.lr_scheduler import CosineLR
from horch.train import manual_seed

from hinas.models.darts.search.pc_darts import Network

from horch.train.cls.metrics import Accuracy
from horch.train.metrics import TrainLoss, Loss

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    manual_seed(args.seed)

    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
    ])

    ds = CIFAR10(root=args.data, train=True, download=True)

    ds_train, ds_search = train_test_split(
        ds, test_ratio=0.5, shuffle=True, random_state=args.seed,
        transform=train_transform, test_transform=train_transform)

    train_queue = DataLoader(
        ds_train, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2)

    valid_queue = DataLoader(
        ds_search, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2)

    set_defaults({
        'relu': {
            'inplace': False,
        },
        'bn': {
            'affine': False,
        }
    })
    model = Network(args.init_channels, args.layers, num_classes=CIFAR_CLASSES)
    criterion = nn.CrossEntropyLoss()

    optimizer_arch = Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay)
    optimizer_model = SGD(
        model.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = CosineLR(
        optimizer_model, float(args.epochs), min_lr=args.learning_rate_min)

    train_metrics = {
        "loss": TrainLoss(),
        "acc": Accuracy(),
    }

    eval_metrics = {
        "loss": Loss(criterion),
        "acc": Accuracy(),
    }

    learner = DARTSLearner(model, criterion, optimizer_arch, optimizer_model, scheduler,
                           train_metrics=train_metrics, eval_metrics=eval_metrics,
                           search_loader=valid_queue, grad_clip_norm=5.0, work_dir='models')

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))
        print(F.softmax(model.betas_normal[2:5], dim=-1))
        # training

        train_acc, train_obj = train(learner, train_queue, epoch)
        logging.info('train_acc %f', train_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(learner, train_queue, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    state = learner._state['train']
    state.update({
        "epoch": epoch,
        "steps": len(train_queue),
    })
    for step, batch in enumerate(train_queue):
        state['step'] = step
        learner.train_arch = epoch >= 15
        learner.train_batch(batch)

        logits = state['y_pred']
        target = state['y_true']
        n = state['batch_size']
        loss = state['loss']
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss, n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
