import torch
import torch.nn as nn
import higher
from higher.patch import monkeypatch as make_functional
import torch.nn.functional as F
from collections import OrderedDict
from torch.func import functional_call
import pdb
import copy
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SPRegularization(nn.Module):
    def __init__(self, source_model: nn.Module, inner_params):
        super(SPRegularization, self).__init__()
        self.source_weight = {}
        self.inner_params = inner_params
        for name, param in source_model.named_parameters():
            if name in inner_params:
                self.source_weight[name] = param.detach()

    def forward(self, target_model):
        batch_loss = 0.0
        for name, param in target_model.named_parameters():
            if name in self.inner_params:
                batch_loss += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return batch_loss

class l1Regularization(nn.Module):
    def __init__(self, source_model: nn.Module, inner_params):
        super(l1Regularization, self).__init__()
        self.source_weight = {}
        self.inner_params = inner_params
        for name, param in source_model.named_parameters():
            if name in inner_params:
                self.source_weight[name] = param.detach()

    def forward(self, target_model):
        batch_loss = 0.0
        for name, param in target_model.named_parameters():
            if name in self.inner_params:
                batch_loss += torch.norm(param - self.source_weight[name], p=1) 
        return batch_loss

class FT(nn.Module):
    def __init__(self, model, checkpoints='', edit_lrs=2e-5, max_edit_steps=100, l2_reg=False, enbale=True):
        super().__init__()
        self.grad_clip =10.
        self.max_edit_steps = max_edit_steps
        self.edit_lrs = edit_lrs
        self.cloc = 1.0
        self.model = model
        self.layer = None
        self.l2_reg = l2_reg

    def assign(self, layer, cloc=1.0):
        self.layer = layer
        self.inner_params = [
          'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer),
          'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer),
          'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer-1),
          'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer-1),
          'backbone.transformer.encoder.layer.{}.ffn.fc2.weight'.format(layer-2),
          'backbone.transformer.encoder.layer.{}.ffn.fc1.weight'.format(layer-2),
        ]

        print(self.inner_params)
        if  self.l2_reg:
            print('layer: {}-{}; l2 loss :{:.3f}'.format(layer-2,layer,cloc))
            self.reg_loss = SPRegularization(self.model, self.inner_params)
        else:
            print('layer: {}-{}; l1 loss :{:.3f}'.format(layer-2,layer,cloc))
            self.reg_loss = l1Regularization(self.model, self.inner_params)
        self.cloc = cloc
        for (n, p) in self.model.named_parameters():
            if n not in self.inner_params:
                p.requires_grad = False

    def forward(self, x):
        pred, _ = self.model(x)
        return pred

    def edit(self, images,target):
        edit_model = copy.deepcopy(self.model).eval()
        opt_params = [{"params": p, "lr": self.edit_lrs} for (n, p) in edit_model.named_parameters() if n in self.inner_params]
        assert len(opt_params) == len(self.inner_params)
        opt_1st = torch.optim.SGD(opt_params, lr= self.edit_lrs)
        opt = torch.optim.RMSprop(opt_params, lr= self.edit_lrs)

        for edit_step in range(self.max_edit_steps):
            output, _ = edit_model(images)
            cls_loss = F.cross_entropy(output, target) 
            acc = output.data.max(1)[1].eq(target.data).sum().item()/target.size(0)

            reg_loss = self.reg_loss(edit_model)
            loss =  cls_loss +  self.cloc*reg_loss
            
            if acc == 1.0 and cls_loss<=0.01:
                break
            loss.backward()
 
            if edit_step == 0:
                opt_1st.step()
                opt_1st.zero_grad()
            else:
                opt.step()
                opt.zero_grad()

        return FT(edit_model, enbale=False)