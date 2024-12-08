import random
import time
from turtle import pd
import warnings
import sys
import argparse
import shutil
import os
import numpy as np
from PIL import Image
import pdb
import torch
import torch.nn as nn
import higher
import torch.backends.cudnn as cudnn
from torch.optim import SGD,Adam
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
import timm
import models as models
from editing.utils import  AverageMeter, ForeverDataIterator, CompleteLogger, Classifier
import copy
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
from pycocotools.coco import COCO
import seaborn as sns
import matplotlib.pyplot as plt
from torch.func import functional_call
from collections import OrderedDict
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mask_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size=224, in_channels=3):
        super(Mask_Embeddings, self).__init__()
        img_size = torch.nn.modules.utils._pair(img_size)
        patch_size = [16, 16]
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                out_channels=768,
                                kernel_size=patch_size,
                                stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, 768))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

    def forward(self, x, masks):
        B = x.shape[0]
        x = self.patch_embeddings(x)
        x = x.flatten(2)  #BxCxM
        x = x.transpose(-1, -2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        new_embeddings = embeddings[0,1:][masks.reshape(-1)==1]
        new_embeddings = torch.cat((embeddings[:, 0:1], new_embeddings.unsqueeze(0)),dim=1)
        return new_embeddings
    
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = 6
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)
        self.out = nn.Linear(768, 768)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)
        self.act_fn =  nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x
    
class Outblock(nn.Module):
    def __init__(self, out_dim=3072):
        super(Outblock, self).__init__()
        self.out = nn.Linear(768, out_dim)
        self.act_out = nn.Sigmoid()
        self.amplify = 3
        self.ep = 1e-5

    def forward(self, x):
        x = self.out(x)
        x = self.act_out(self.amplify*x)
        x = torch.clamp(x,min=self.ep, max=1-self.ep)
        return x     
    
class HyperNetwork_mask(nn.Module):
    def __init__(self,layers=6, blocks=5):
        super(HyperNetwork_mask , self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 6, 768))
        self.layer = nn.ModuleList()
        for _ in range(blocks):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

        self.out_layer = nn.ModuleList()
        for _ in range(layers):
            layer = Outblock()
            self.out_layer.append(copy.deepcopy(layer))
        self.learn_masks = nn.Parameter(2*torch.rand(6,3072)-1)

    def forward(self, x, debug=False):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for layer_block in self.layer:
            x = layer_block(x)
        mask_output = []
        for i, out_layer_block in enumerate(self.out_layer):
            mask_output.append(out_layer_block(x[:,i:i+1]))
        mask_output = torch.cat(mask_output,dim=1)
        return mask_output

class inner_masks(nn.Module):
    def __init__(self,batch_size):
        super(inner_masks , self).__init__()
        self.learn_masks = nn.Parameter(2*torch.rand(batch_size,6,3072)-1)

    def forward(self, scale):
        return F.sigmoid(scale*self.learn_masks)

class VariationalMasks(nn.Module):
    """
    """
    def __init__(self, model, layer=9, lr=1e-4, elr=1e-3, max_edit_steps=1, max_out_steps=1, sparsity_weights=0.0, blocks=5):
        super(VariationalMasks, self).__init__()
        self.pretrained_model = model.eval()
        self.edit_model = copy.deepcopy(model.eval())

        self.edit_layers = layer
        self.edit_lr = elr
        self.max_edit_steps = max_edit_steps
        self.max_out_steps = max_out_steps
        self.lamda = sparsity_weights
        self.edit_parameters = self.get_edit_parameters(self.edit_layers)
        self.hyper_network = HyperNetwork_mask(len(self.edit_parameters), blocks=int(blocks)).cuda()
        self.reg = False
        self.opt = torch.optim.RMSprop(self.hyper_network.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, 2e4) 
        self.T = 5.
        self.alpha = 0.75
        self.vis_iter = 0
        self.sparsity_loss = AverageMeter('sparsity_loss', ':3.5f')
        self.bce_loss = AverageMeter('bce_loss', ':3.5f')
        self.out_kl_loss = AverageMeter('out_kl_loss', ':3.5f')
        self.mask_diff = AverageMeter('mask_diff', ':3.5f')
        self.label_out_kl = AverageMeter('label_out_kl', ':3.5f')
        self.sparsity = AverageMeter('sparsity', ':3.5f')
        self.label_sparsity = AverageMeter('label_sparsity', ':3.5f')

        self.total_loss = 0.
        self.masks_embeddings = Mask_Embeddings()
        self.masks_embeddings.load_state_dict(model.backbone.transformer.embeddings.state_dict(), strict=False)
        self.masks_embeddings.position_embeddings = model.backbone.transformer.embeddings.position_embeddings
        self.masks_embeddings.cls_token = model.backbone.transformer.embeddings.cls_token
        self.masks_embeddings.eval()

    def get_edit_parameters(self,layer):
        edit_parameters = [
          'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer),
          'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer),
          'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer-1),
          'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer-1),
          'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer-2),
          'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer-2),
        ]
        return edit_parameters
    def reset_meter(self):
        self.sparsity.reset()
        self.mask_diff.reset()
        self.label_out_kl.reset()   #label_mask_diff
        self.bce_loss.reset()
        self.out_kl_loss.reset()
        self.label_sparsity.reset()
        
    def train_hypernetwork(self, original_images, mixed_images, target):
        edited_model = self.edit_model    
        ### get image features
        batch_size = mixed_images.size(0)
        with torch.no_grad():
            ori_output, _ = self.pretrained_model(original_images)
            bias_output, image_features = self.pretrained_model(mixed_images, False)
            ori_output = ori_output.detach().clone().requires_grad_(False)
        self.pretrained_model.zero_grad()
        combination = image_features[:,0:1].detach().clone().requires_grad_(False)

        ###generate masks via  hypernetwork
        masks = self.hyper_network(combination)   
        masks = masks.reshape(batch_size, len(self.edit_parameters), 3072)   
        total_sparsity =  1. - masks.mean().item()
        self.sparsity.update(total_sparsity)
        sparse_penalty = self.sparse_penalty(masks)
        self.sparsity_loss.update(sparse_penalty.item())

        ###Decouple training   
        learn_masks_module = inner_masks(batch_size).to(device)
        opt = torch.optim.RMSprop(learn_masks_module.parameters(), lr=1e-1) ##SGD RMSprop
        for steps in range(self.max_out_steps):
            total_kl_loss = 0.0
            learn_masks = learn_masks_module(10)
            for i in range(batch_size):
                edited_model_state = self.get_params(edited_model)   ###init model
                delta_model_state = self.get_delta_params(edited_model)  ###zero model
                # Inner loop
                for _ in range(self.max_edit_steps):
                    params = self._interpolate_params(edited_model_state, delta_model_state, learn_masks, i)
                    edited_output, _ =  functional_call(edited_model, params, mixed_images[i:i+1])
                    loss = self.T*self.T*F.kl_div(F.log_softmax(edited_output/self.T,dim=-1),
                                F.softmax(ori_output[i:i+1]/self.T,dim=-1),reduction='batchmean')
                    grads = torch.autograd.grad(loss, delta_model_state.values(),
                                                create_graph=False, #True
                                                retain_graph=False,
                                                allow_unused=True)
                    ind = 0
                    for  (n,  p), grad in zip (list(delta_model_state.items()), grads):
                        if n in self.edit_parameters:
                            if n in self.edit_parameters:
                                if 'fc1' in n:
                                    p -= self.edit_lr * grad / (learn_masks[i, ind].detach())[:,None]
                                else:
                                    p -=  self.edit_lr * grad / (learn_masks[i, ind].detach())[None,:]
                                ind += 1
                # outer loop
                params = self._interpolate_params(edited_model_state, delta_model_state, learn_masks, i)
                edited_output, _ =  functional_call(edited_model, params, mixed_images[i:i+1] )
                loss = self.T*self.T*F.kl_div(F.log_softmax(edited_output/self.T,dim=-1),
                            F.softmax(ori_output[i:i+1]/self.T,dim=-1),reduction='batchmean')
                total_kl_loss += loss
            sparse_penalty = self.sparse_penalty(learn_masks)
            # test codes
            # temp_masks = learn_masks.detach().cpu()
            # mask_diff = (((temp_masks-temp_masks.flip(0)).abs()>0.2).sum()/batch_size)
            # print('Outer step {}: kl_loss: {:.3f}, sparsity: {:.3f} mask diff: {:.1f} '.format(steps, total_kl_loss.item()/batch_size, 1.-learn_masks.mean().item(), mask_diff))
            total_loss = total_kl_loss + self.lamda*sparse_penalty
            total_loss.backward()
            opt.step()
            opt.zero_grad()
        self.label_out_kl.update(total_kl_loss.item()/batch_size)
        label_masks = learn_masks_module(3).clone().detach().to(masks.device)
        
        # test codes, set validate true
        validate = False
        if validate:
            validate_masks = masks.clone().detach()
            total_loss = 0.0
            for i in range(batch_size):
                ### inner loop for each sample
                edited_model_state = self.get_params(edited_model)   ###init model
                delta_model_state = self.get_delta_params(edited_model)  ###zero model
                # inner loop
                for t in range(self.max_edit_steps):
                    params = self._interpolate_params(edited_model_state, delta_model_state, validate_masks, i)
                    edited_output, _ =  functional_call(edited_model, params, mixed_images[i:i+1]) 
                    loss = self.T*self.T*F.kl_div(F.log_softmax(edited_output/self.T,dim=-1),
                                F.softmax(ori_output[i:i+1]/self.T,dim=-1),reduction='batchmean')
                    grads = torch.autograd.grad(loss, delta_model_state.values(),
                                                create_graph=False, #True
                                                retain_graph=False,
                                                allow_unused=True)
                    ind = 0
                    for  (n,  p), grad in zip (list(delta_model_state.items()), grads):
                        if n in self.edit_parameters:
                            if n in self.edit_parameters:
                                if 'fc1' in n:
                                    p -= self.edit_lr * grad / (validate_masks[i, ind].detach())[:,None]
                                else:
                                    p -=  self.edit_lr * grad / (validate_masks[i, ind].detach())[None,:]
                                ind += 1
                # outer loop
                params = self._interpolate_params(edited_model_state, delta_model_state, validate_masks, i)

                edited_output, _ =  functional_call(edited_model, params, mixed_images[i:i+1]) 
                kl_loss = self.T*self.T*F.kl_div(F.log_softmax(edited_output/self.T,dim=-1),
                            F.softmax(ori_output[i:i+1]/self.T,dim=-1),reduction='batchmean')
                total_loss += kl_loss
                self.out_kl_loss.update(kl_loss.item())
            print('hyper masks: kl_loss: {:.5f}, sparsity: {:.3f} '.format(total_loss.item()/batch_size, 1.-validate_masks.mean().item()))
            pdb.set_trace()
        bce_loss = F.binary_cross_entropy(masks, label_masks)
        self.bce_loss.update(bce_loss.item())
        self.total_loss += 0.1*bce_loss
        self.total_loss.backward()
        
        self.opt.step()
        self.opt.zero_grad()
        self.total_loss = 0.
        self.label_sparsity.update(1. - label_masks.mean().item())
        for _ in range(3):
            masks = self.hyper_network(combination)   
            masks = masks.reshape(batch_size, len(self.edit_parameters), 3072)   
            bce_loss = F.binary_cross_entropy(masks, label_masks)
            loss = 0.1*bce_loss   
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        temp_masks = masks.detach().cpu()
        self.mask_diff.update(((temp_masks-temp_masks.flip(0)).abs()>0.2).sum()/batch_size)

    def sparse_penalty(self, log_alphas):
        l1_regularization  = torch.norm(log_alphas, 1)
        normalizer = log_alphas.numel() 
        return l1_regularization/normalizer
    
 
    def get_delta_params(self, temp):
        return {n: (p-p).clone().requires_grad_(True) for n, p in temp.named_parameters()}
    
    def get_params(self, temp):
        return {n: p.clone().requires_grad_(True) for n, p in temp.named_parameters()} 
    
    def _interpolate_params(self, edited_model_state, delta_model_state, z_masks, i):
        params = OrderedDict()
        ind = 0
        for (name, param), (delta_name, delta_param) in zip(edited_model_state.items(),delta_model_state.items()) :
            if name in self.edit_parameters:
                if 'fc1' in name:
                    params[name] = param.detach() + delta_param*z_masks[i, ind][:,None]
                else:
                    params[name] = param.detach() + delta_param*z_masks[i, ind][None,:]
                ind += 1
            else:
                params[name] = param.detach()
        return params
    
    def save_para(self,dirs):
        torch.save(self.hyper_network.state_dict(), dirs)
    def save_model(self,dirs):
        torch.save(self.hyper_network, dirs)

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

    # Data loading code
    train_whole_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.25, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    num_classes = 1000
    pretrain_train_set = torchvision.datasets.ImageFolder(root=os.path.join(args.pre_root,'train'), transform=train_whole_transform)
    batch_size = args.batch_size
    print('dataset size: {}, batch size: {}'.format(len(pretrain_train_set), batch_size))
    pretrain_train_loader = torch.utils.data.DataLoader(pretrain_train_set, batch_size=2*batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    pretrain_train_loader = ForeverDataIterator(pretrain_train_loader)

    backbone = models.__dict__[args.arch](pretrained=True, n_classes=num_classes).to(device)
    classifier = Classifier(backbone=backbone, num_classes=num_classes, finetune=True).to(device)
    classifier.head = copy.deepcopy(backbone.pred_head)

    hyper_masks = VariationalMasks(
        model = classifier,
        layer = args.layer,
        lr = args.lr,
        elr = args.elr,
        max_edit_steps =  args.mes,
        max_out_steps = args.mos,
        sparsity_weights = args.sw,
        blocks = args.blocks,
    )
    iterations = 10000
    hyper_masks = hyper_masks.to(device)
    if args.phase == 'train':
        hyper_masks.hyper_network.train()
        hyper_masks.pretrained_model.eval()
        hyper_masks.edit_model.eval()
        for iter in range(iterations):
            images, target = next(pretrain_train_loader)
            mixed_images, concept_regions = cutmix(images)
            hyper_masks.train_hypernetwork(images[0:batch_size].to(device), mixed_images.to(device), target[0:batch_size].to(device))
            if (iter+1) % args.print_freq == 0 :
                print('iterations: {}, sparsity: {sparsity.avg:.5f}, out_kl_loss: {out_kl_loss.avg:.5f},  label_sparsity: {label_sparsity.avg:.5f}, label_out_kl_loss: {label_out_kl.avg:.5f}, bce_loss: {bce_loss.avg:.5f}, mask_diff: {mask_diff.avg:.1f}'
                        .format(iter+1, sparsity=hyper_masks.sparsity, out_kl_loss=hyper_masks.out_kl_loss, label_sparsity=hyper_masks.label_sparsity,\
                        bce_loss=hyper_masks.bce_loss, mask_diff=hyper_masks.mask_diff, label_out_kl=hyper_masks.label_out_kl) )
                hyper_masks.reset_meter()
            hyper_masks.lr_scheduler.step()
            if (iter+1) % 500 == 0 and not args.debug:
                hyper_masks.save_para(logger.get_checkpoint_path(str(iter+1)))
        hyper_masks.save_para(logger.get_checkpoint_path('latest'))
    logger.close()


def cutmix(images):
    image_list = []
    concept_list = []
    batch_size = images.size(0)//2
    for i in range(batch_size):
        mix_ima = images[i].clone() 
        ima_2 = images[i+batch_size]
        #### sample patches
        img_h, img_w = (14,14)
        cut_h, cut_w = np.random.randint(3,9, size=2)
        yl = np.random.randint(0, img_h-cut_h)
        xl = np.random.randint(0, img_w-cut_w)
        yh = yl+cut_h
        xh = xl+cut_w
        concept_masks = torch.zeros(img_h, img_w)
        concept_masks[yl:yh,xl:xh] = 1.
        mix_ima[:, yl*16:yh*16, xl*16:xh*16] = ima_2[:, yl*16:yh*16, xl*16:xh*16]
        image_list.append(mix_ima.unsqueeze(0))
        concept_list.append(concept_masks)
    return torch.cat(image_list,dim=0), concept_list



if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='Baseline for Finetuning')
    # dataset parameters
    parser.add_argument('--pre_root', default='/dataset/imagenet1k/')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_b_16_224_1k',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: vit_b_16_224_1k)')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--mes', '--max-edit-steps', default=5, type=int)
    parser.add_argument('--mos', '--max-out-steps', default=10, type=int)
    parser.add_argument('--elr',  default=1e-3, type=float)
    parser.add_argument('--sw','--sparsity-weights',  default=1.0, type=float)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='./log',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train',
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--pre-weights', action="store_true",
                        help='add weights on some classes')
    parser.add_argument('--layer', default=9, type=int)
    parser.add_argument('--blocks', default=5, type=int)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--pretrained', action="store_true")
    args = parser.parse_args()
    main(args)

