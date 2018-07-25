# coding=utf8

import argparse

parser = argparse.ArgumentParser(description='medical caption GAN')

parser.add_argument(
        '--model',
        '-m',
        metavar='mynet',
        type=str,
        default='mynet',
        help='classification model'
        )
parser.add_argument(
        '--data-dir',
        '-d',
        metavar='../data/',
        type=str,
        default='../data/',
        help='data directory'
        )
parser.add_argument(
        '--phase',
        '-p',
        metavar='PHASE',
        type=str,
        default='train',
        help='train or test'
        )
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=16,
        help='batch size'
        )
parser.add_argument('-j',
        '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--epochs',
        default=600,
        type=int,
        metavar='N',
        help='number of total epochs to run')
parser.add_argument('--save-freq',
        default='5',
        type=int,
        metavar='S',
        help='save frequency')
parser.add_argument('--save-pred-freq',
        default='10',
        type=int,
        metavar='S',
        help='save pred clean frequency')
parser.add_argument('--val-freq',
        default='5',
        type=int,
        metavar='S',
        help='val frequency')
parser.add_argument('--save-dir',
        default='default',
        type=str,
        metavar='S',
        help='save dir')
parser.add_argument('--resume',
        default='',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument('--debug',
        default=0,
        type=int,
        metavar='S',
        help='debug')
parser.add_argument('--input-filter',
        default=7,
        type=int,
        metavar='S',
        help='val frequency')
parser.add_argument('--use-gan',
        default=0,
        type=int,
        metavar='S',
        help='use GAN')
parser.add_argument('--write-pred',
        default=-1,
        type=int,
        metavar='S',
        help='writ predictions')
parser.add_argument('--test',
        default=0,
        type=int,
        metavar='S',
        help='test phase')
parser.add_argument('--training-time',
        default=1,
        type=int,
        metavar='S',
        help='training time')
args = parser.parse_args()
