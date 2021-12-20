import torch
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MeshedDecoder, ScaledDotProductAttentionMemory,VisualPlusSemeticEncoder
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from info import  args

def build_model_baseline():
    pass

def build_model_cross_att():
    pass

def get_param():
    pass

if __name__ == '__main__':
    pass