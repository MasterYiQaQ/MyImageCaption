import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
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
from info import args
from models.transformer.VisualPlusSemetic import VisualPlusSemeticEncoder
from models.transformer.geom import ratio_geomertical

bbox = torch.randn(10,50,4)
bbox = ratio_geomertical(bbox)
print(bbox.shape)

# image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
# # Pipeline for text
# text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
#                            remove_punctuation=True, nopoints=False)
#
# dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
# train_dataset, val_dataset, test_dataset = dataset.splits
# if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
#     print("Building vocabulary")
#     text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
#     pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
# else:
#     text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

# encoder = VisualPlusSemeticEncoder()
# dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
# dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
# dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,num_workers=args.workers)
# for it, (detections,captions) in enumerate(dict_dataloader_train):
#     images = detections[0]
#     bbox = detections[1]
#     labels = detections[2]
#     encoder(images,bbox,labels)
#     continue
    # print(detections.shape)
    # print(captions.shape)