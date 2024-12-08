import random
import time
import warnings
import sys
import argparse
import shutil
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision
import torch.nn.functional as F
import copy
import timm
import models as models
import editing as edits_alg
from editing.utils import AverageMeter, CompleteLogger,  all_wnids, imagenet_a_wnids,imagenet_r_wnids, Classifier
from torch.utils.data.dataset import ConcatDataset
import pdb
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseTrainer:
    def __init__(self, alg, args, pretrain_loader, problem_loader, locality_loder, logger):
        self.alg = alg
        try:
            self.original_model = copy.deepcopy(alg)
        except:
            self.original_model = alg
        self.config = args
        self.pretrain_loader = pretrain_loader
        self.problem_loader = problem_loader
        self.locality_loder = locality_loder
        self.mask = None
        self.logger = logger
        # self.search_masks = []

    def run(self, online=False):
        sr = AverageMeter('sr', ':3.5f')
        gr = AverageMeter('gr', ':3.5f')
        lr = AverageMeter('lr', ':3.5f')

        for i, (images, target) in enumerate(self.problem_loader[0]):
            sr_i, gr_i, lr_i = self.edit_step_val(images, self.problem_loader[1], target)
            sr.update(sr_i, 1)
            lr.update(lr_i, 1)
            gr.update(gr_i, 1)
        print('* sr {sr.avg:.3f} gr {gr.avg:.5f} lr {lr.avg:.5f}  '
              .format(sr=sr, gr=gr, lr=lr))
        # self.search_masks = np.array(np.stack(self.search_masks))

    def edit_step_val(self, images, gr_loaders,  target):
        self.alg.eval()
        self.original_model.eval() 
        label = target.to(device)
        x = images.to(device)
        edited_model = self.alg.edit(x, label)

        # self.search_masks.append(self.alg.store_masks_out)
        with torch.no_grad():
            output = edited_model(x)
            edited_preds = output.data.max(1)[1]  
            sr = edited_preds.eq(label.data).sum().item()/label.size(0)*100

            ori_output = self.original_model(x)
            ori_preds = ori_output.data.max(1)[1]  
            num_all = 0
            num_correct = 0
            for _, (images, target) in enumerate(gr_loaders):
                images = images.to(device)
                target = target.to(device)
                output = edited_model(images)
                edited_preds = output.data.max(1)[1]         
                num_correct += edited_preds.eq(target).sum().item()
                num_all += target.size(0)
            gr = (num_correct-1)/num_all*100
        
            num_all = 0
            num_correct = 0
            for j, (images, target) in enumerate(self.locality_loder):
                images = images.to(device)
                target = target.to(device)
                output = self.original_model(images)
                original_preds = output.data.max(1)[1]
                edit_output = edited_model(images)
                edited_preds = edit_output.data.max(1)[1]         
                num_correct += original_preds.eq(edited_preds).sum().item()
                num_all += target.size(0)
            lr = num_correct/num_all*100

        return sr, gr, lr

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dirs, transform, label):
        super(Dataset, self).__init__()
        self.transform = transform
        self.dir = dirs
        self.img_dir = os.listdir(dirs)
        self.labels = label

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item
    
    def load_item(self, index):
        img = Image.open(os.path.join(self.dir, self.img_dir[index])).convert('RGB')
        img = self.transform(img)
        label = int(self.labels)
        return img, label
    
def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    ###### data loader
    pretrain_loader = None       
    num_classes = 1000   
    
    sensitive_images_set = Dataset(dirs='./Sensitive-Images/', 
                                transform=val_transform,
                                label = int(0))
    sensitive_images_loader = DataLoader(sensitive_images_set, batch_size=64, shuffle=False, num_workers=2)

    backbone = models.__dict__[args.arch](pretrained=True, n_classes=num_classes)
    classifier = Classifier(backbone=backbone, num_classes=num_classes, finetune=True).to(device)
    classifier.head = copy.deepcopy(backbone.pred_head)

    root = args.root
    sub_classes = os.listdir(root)
    for sub_class in sub_classes:
        print('class: ' + sub_class)
        root_dir = os.path.join(root, sub_class)
        part_image_net = Dataset(dirs=root_dir, 
                                transform=val_transform,
                                label = int(sub_class.split('_')[0]))
        select_image =  select_data(part_image_net, copy.deepcopy(classifier))
        part_image_net = Subset(part_image_net, select_image)
        print(len(part_image_net))
        part_image_loader_gr = DataLoader(part_image_net, batch_size=64, shuffle=False, num_workers=1)
        part_image_loader_edit = DataLoader(part_image_net, batch_size=1, shuffle=False, num_workers=1)
        part_image_loader = [part_image_loader_edit, part_image_loader_gr]
        _iters =  args.max_iters
        
        if args.alg in ['HPRD']:
            print(args.checkpoints)
            layer=9
            for sparsity in  [0.75, 0.5, 0.25, 0.1, 0.05]:
                model = edits_alg.__dict__[args.alg](blocks=args.blocks, model=copy.deepcopy(classifier), checkpoints=args.checkpoints, edit_lrs=args.edit_lrs,layer=layer, max_edit_steps=_iters, l2_reg=args.l2reg).to(device)
                model.sparsity = sparsity
                print('sparsity: {:.2f}, hp blocks {:.1f}'.format(sparsity, args.blocks) )
                triner = BaseTrainer(model, args, pretrain_loader, part_image_loader, sensitive_images_loader, logger)
                triner.run()

        elif args.alg in ['FT']:
            ranges =  [0.0, 0.001, 0.005,  0.01, 0.025, 0.05,0.1] 
            ranges.reverse()
            for clo in ranges:
                model = edits_alg.__dict__[args.alg](model=copy.deepcopy(classifier), checkpoints='', edit_lrs=args.edit_lrs, max_edit_steps=_iters, l2_reg=args.l2reg).to(device)
                model.assign(9,clo) 
                triner = BaseTrainer(model, args, pretrain_loader, part_image_loader, sensitive_images_loader, logger)
                triner.run()
        
        else:
            model = edits_alg.__dict__[args.alg](model=copy.deepcopy(classifier), edit_lrs=args.edit_lrs, max_edit_steps=_iters).to(device)
            triner = BaseTrainer(model, args, pretrain_loader, part_image_loader, sensitive_images_loader, None, logger)
            triner.run()
    logger.close()

def select_data(part_image_net, classifier):
    classifier.eval()
    classifier = classifier.to(device)
    load_data = DataLoader(part_image_net, batch_size=1, shuffle=False, num_workers=1)
    select_list = []
    for j, (images, target) in enumerate(load_data):
        images = images.to(device)
        target = target.to(device)
        output,_ = classifier(images)
        edited_preds = output.data.max(1)[1]         
        if edited_preds.eq(target).sum().item()==0:
            select_list.append(j)
    print('select {} from {}'.format(len(select_list), len(part_image_net)))
    return select_list


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Baseline for Editing')
    parser.add_argument('--root', metavar='DIR', default='./Natural-Image-Subset',
                        help='root path of dataset')
    parser.add_argument('--alg', metavar='ALG', default='FT')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_b_16_224_1k',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--l2reg', action="store_true")
    parser.add_argument('--edit-lrs', default=1e-4, type=float)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='test', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--max-iters', default=10, type=int)
    parser.add_argument('--rangel', action="store_true")
    parser.add_argument('--factor', default=1, type=int, help='hidden dims')
    parser.add_argument('--checkpoints', default='./log/checkpoints/7000.pth')
    parser.add_argument('--blocks', default=3, type=int, help='blocks in hp')

    args = parser.parse_args()
    main(args)

