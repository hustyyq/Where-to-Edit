import torch
import torch.nn as nn
import higher
from higher.patch import monkeypatch as make_functional
import torch.nn.functional as F
import numpy as np
import time
from .utils import _inner_params
import copy
import os
import pdb
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HPRD(nn.Module):
    def __init__(self, model, checkpoints='', edit_lrs=1e-5, max_edit_steps=100,l2_reg=False,layer=9, blocks=5, enable_hyper_network=True):
        super().__init__()
        self.max_edit_steps = max_edit_steps
        self.edit_lrs = edit_lrs
        self.model = model
        self.layer = layer
        self.blocks = blocks
        self.inner_params = [
        'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer),
        'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer),
        'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer-1),
        'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer-1),
        'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer-2),
        'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer-2),
        ]
        self.sparsity = 0.1
        if enable_hyper_network:
            mask_hypernetwork = VariationalMasks(model, blocks=self.blocks)
            mask_hypernetwork.hyper_network.load_state_dict(torch.load(checkpoints), strict=True)
            self.mask_hypernetwork = mask_hypernetwork.eval().cuda()
            self.query_model = copy.deepcopy(model.backbone)
            self.query_model.eval()
            print(self.inner_params)
        self.masks = None
        self.store_masks = []

    def get_impt(self, images, target):
        with torch.no_grad():
            query_f =  self.query_model(images, False) ###image feature  
            combination = query_f[:, 0:1].detach().clone().requires_grad_(False)
            masks, save_masks = self.mask_hypernetwork.generate_masks(combination, self.sparsity)
            # self.store_masks.append(save_masks.detach().cpu().numpy())

        self.masks = masks.to(device)

    def forward(self, x):
        pred, _ = self.model(x)
        return pred

    def edit(self, images, target):
        edit_model = self.model.eval()
        if not isinstance(edit_model, higher.patch._MonkeyPatchBase):
            edit_model = make_functional(self.model, track_higher_grads=True)
        opt_params = [{"params": p, "lr": self.edit_lrs} for (n, p) in edit_model.named_parameters() if n in self.inner_params]
        assert len(opt_params) == len(self.inner_params)
        opt = torch.optim.RMSprop(opt_params, lr=self.edit_lrs)

        with torch.no_grad():
            self.get_impt(images, target)
        for steps in range(self.max_edit_steps):
            output, _ = edit_model(images)
            cls_loss = F.cross_entropy(output, target) 
            loss =  cls_loss 
            acc = output.data.max(1)[1].eq(target.data).sum().item()/target.size(0)
            if acc == 1.0 and cls_loss<=0.01:
                break
            loss.backward()

            for j in range(self.layer-2,self.layer+1):
                edit_model.backbone.transformer.encoder.layer[j].ffn.fc1.weight.grad *= self.masks[2*(j-self.layer+2)][:, None]
                edit_model.backbone.transformer.encoder.layer[j].ffn.fc2.weight.grad *= self.masks[2*(j-self.layer+2)+1][None,:]
            opt.step()
            opt.zero_grad()
        
        return HPRD(edit_model, enable_hyper_network=False)

class Old_Mask_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size=224, in_channels=3):
        super(Old_Mask_Embeddings, self).__init__()
        self.hybrid = None
        img_size = torch.nn.modules.utils._pair(img_size)
        patch_size = [16, 16]
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                out_channels=768,
                                kernel_size=patch_size,
                                stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, 768))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.dropout = nn.Dropout(0.1)
        self.stride = 3

    def forward(self, x, mask=None):
        B = x.shape[0]
        x = self.patch_embeddings(x) #*mask[:,None,:,:]
        x = x.flatten(2)  #BxCxM
        x = x.transpose(-1, -2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        if mask is not None:
            new_embeddings = embeddings[0,1:][mask.reshape(-1)==1]
            new_embeddings = torch.cat((embeddings[:, 0:1], new_embeddings.unsqueeze(0)),dim=1)
            return new_embeddings
        else:
            return embeddings



from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import math

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
    
class VariationalMasks(nn.Module):
    """
    Implements the hyper-net for param or gradient masks whose distribution is a streched hard-concrete.
    """
    def __init__(self,model, layer=9, vis=False,blocks=5):
        super(VariationalMasks, self).__init__()
        self.edit_layers = layer
        self.blocks = blocks
        self.edit_parameters = self.get_edit_parameters(self.edit_layers)
        self.hyper_network = HyperNetwork_mask(len(self.edit_parameters),  blocks)
        self.masks_embeddings = Mask_Embeddings()
        self.pretrained_model = copy.deepcopy(model.eval())
        self.masks_embeddings.load_state_dict(self.pretrained_model.backbone.transformer.embeddings.state_dict(), strict=False)
        self.masks_embeddings.position_embeddings = self.pretrained_model.backbone.transformer.embeddings.position_embeddings
        self.masks_embeddings.cls_token = self.pretrained_model.backbone.transformer.embeddings.cls_token
        
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
    
    def generate_masks(self, x, sparsity): 
        masks = self.hyper_network(x)
        b = x.size(0)
        masks = masks.reshape(b,len(self.edit_parameters), 3072).mean(0)
        v, _ = torch.topk(masks.reshape(-1), int(sparsity*masks.numel()) )
        z_masks = torch.greater(masks, v[-1]).float() # convert to {0,1}

        return z_masks, masks

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
        self.ep = 1e-7

    def forward(self, x):
        x = self.out(x)
        x = self.act_out(self.amplify*x)
        x = torch.clamp(x,min=self.ep, max=1-self.ep)
        return x     
    
class HyperNetwork_mask(nn.Module):
    def __init__(self,layers, blocks=5):
        super(HyperNetwork_mask , self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 6, 768))
        self.layer = nn.ModuleList()
        for _ in range(blocks):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))
        self.out_layer = nn.ModuleList()
        for _ in range(6):
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