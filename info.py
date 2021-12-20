import argparse
import torch

parser = argparse.ArgumentParser(description='MyImageCaption_BaseLine')
parser.add_argument('--exp_name', type=str, default='MyImageCaption')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--m', type=int, default=40)
parser.add_argument('--head', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--resume_last', action='store_true')
parser.add_argument('--resume_best', action='store_true')
parser.add_argument('--features_path', type=str, default='/home/datasets/coco_detections.hdf5')
parser.add_argument('--annotation_folder', type=str, default='/home/User/wangyi/annontions')
parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
parser.add_argument('--ckpt_local', type=str, default='/home/User/wangyi/MyImageCaption/saved_models/MyImageCaption_ALL_best.pth')
args = parser.parse_args()