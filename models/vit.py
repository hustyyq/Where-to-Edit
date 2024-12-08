import copy
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
# from .vit_backbone import ResNetV2
from . import config as configs
# import config as configs


__all__ = ['vit_b_16_224_1k', 'vit_s_16_224_1k',
            'vit_b_16_224_1k_lora', 'vit_s_16_224_1k_lora']


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_features = config.hidden_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.attention_probs = None
    #     self.gradients = None

    # def save_gradient(self, grad):
    #     self.gradients = grad

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
        # self.attention_probs = attention_scores.clone()  #if self.vis else None
        attention_probs = self.softmax(attention_scores)
        # attention_probs.register_hook(self.save_gradient)
        attention_probs = self.attn_dropout(attention_probs)
        self.attention_probs = attention_probs.clone()  #if self.vis else None

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, self.attention_probs 
    
    def get_attention_map(self):
        return self.attention_probs

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.act = None
        self.val = None
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.act_fn(x)
        self.act = x.clone()
        x = self.dropout(x)
        if mask is not None:
            # x[:,0,:] = x[:,0,:]*mask
            x[:,:,mask[0]] = mask[1][:,:,mask[0]]
        x = self.fc2(x)
        self.val = x.clone()
        x = self.dropout(x)
        return x
    
    def get_act(self):
        act = self.act.clone()
        self.act = None
        return act
    
    def get_val(self):
        val = self.val.clone()
        self.val = None
        return val
    
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                    out_channels=config.hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        elif config.split == 'non-overlap':
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                    out_channels=config.hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)
            self.hybrid = False
        elif config.split == 'overlap':
            patch_size = _pair(config.patches["size"])
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
            self.hybrid = False


        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def prompt_forward(self, x, instance_prompt):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = torch.cat((embeddings, instance_prompt), dim=1)
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.out_val = []
        self.out_x = None

    def forward(self, x, mask=None):
        self.out_val = []
        if mask is None:
            # self.out_val.append(x.clone())
            h = x
            x = self.attention_norm(x)
            x, weights = self.attn(x)
            # self.out_val.append(x.clone())
            x = x + h
            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            # self.out_val.append(x.clone())
            # print(x[:,0].max())
            x = x + h
            self.out_x  = x
            return x, weights
        else:
            return self.mask_forward(x, mask)
    
    def get_val(self):
        out_val = self.out_val
        return out_val
    
    def get_out(self):
        return self.out_x 

    def mask_forward(self, x, masks):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x,masks)
        # x[:,0,:] = x[:,0,:]*masks #[:,None,:]
        x = x + h
        return x, weights
    
    def get_atten(self, x):
        h = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        return x 
    
    def forward_serve_atten(self, x, atten_cor):
        h = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        x[:,0,:] = atten_cor
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x
    
    def get_mlp(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x 
    
    def forward_serve_mlp(self, x, mlp_cor):
        h = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x[:,0,:] = mlp_cor
        x = x + h
        return x
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.emb_in = None

    def forward(self, hidden_states, mask=None):
        self.emb_in = hidden_states.detach().clone()
        if mask is None:
            attn_weights = []
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded, attn_weights
        else:
            return self.mask_forward(hidden_states, mask)
        
    def mask_forward(self, hidden_states, soft_masks):
        attn_weights = []
        for i, layer_block in enumerate(self.layer):
            if i in [5,6,7,8]:
                hidden_states, _ = layer_block(hidden_states, soft_masks[:,i-5,:])
            else:
                hidden_states, _= layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, n_classes=1000, vis=False):
        super(VisionTransformer, self).__init__()
        self.classifier = config.classifier
        self._out_features = config.hidden_size
        self.transformer = Transformer(config, img_size, vis)
        self.pred_head = Linear(config.hidden_size, n_classes)

    def reset_parameters(self, state_dict = None):
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x, baseline=True):
        x, attn_weights = self.transformer(x)
        if baseline:
            return x[:, 0]
        else:
            return  x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features


    def load_from(self, weights):
        with torch.no_grad():
            self.pred_head.weight.copy_(np2th(weights["head/kernel"]).t())
            self.pred_head.bias.copy_(np2th(weights["head/bias"]).t())
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        x = self.w(x) + (self.alpha // self.r) * self.w_b(self.w_a(x))
        return x

class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, vit_model, r: int, alpha: int, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        assert alpha > 0
        base_vit_dim = vit_model.transformer.encoder.layer[0].attn.query.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.encoder.layer)))
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.encoder.layer):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.query
            w_v_linear = blk.attn.value
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.query = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q, r, alpha)
            blk.attn.value = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v, r, alpha)
            # print(w_a_linear_q.weight.numel())
        self.reset_lora_parameters()
        self.lora_vit = vit_model
        self.out_features = vit_model._out_features
        self.pred_head =vit_model.pred_head
    def reset_lora_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
    def forward(self, x, baseline=True):
        x = self.lora_vit(x,baseline=True)

        return  x
        
CONFIGS = {
    'ViT_B_16_224_1k': configs.get_b16_config(),
    'ViT_S_16_224_1k': configs.get_s16_config(),
    'testing': configs.get_testing(),
}

Pretrained = {
        'ViT_B_16_224_1k': "vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
        'ViT_S_16_224_1k':  "vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
}



def vit_b_16_224_1k(pretrained=True, image_size=224,  n_classes=1000,  vis=False):
    config = CONFIGS['ViT_B_16_224_1k']
    config.transformer.dropout_rate = 0.0
    model = VisionTransformer(config,img_size=image_size, n_classes=n_classes, vis=vis)
    if pretrained:
        try:
            model.load_from(np.load(Pretrained['ViT_B_16_224_1k']))
            print("Load pre-trained ViT"+Pretrained['ViT_B_16_224_1k'])
        except:
            print("Unable to load pre-trained model, please put the file in " + Pretrained['ViT_B_16_224_1k'])
    return model


def vit_b_16_224_1k_lora(pretrained=True, image_size=224,  n_classes=1000,  r = 4,alpha=4,lora_layer=None, vis=False):
    config = CONFIGS['ViT_B_16_224_1k']
    config.transformer.dropout_rate = 0.0
    model = VisionTransformer(config,img_size=image_size, n_classes=n_classes, vis=vis)
    if pretrained:
        try:
            model.load_from(np.load(Pretrained['ViT_B_16_224_1k']))
            print("Load pre-trained ViT"+Pretrained['ViT_B_16_224_1k'])
        except:
            print("Unable to load pre-trained model, please put the file in " + Pretrained['ViT_B_16_224_1k'])
    # lets freeze first
    mode = LoRA_ViT(model, r, alpha, lora_layer)
    return model


def vit_s_16_224_1k(pretrained=True, image_size=224,  n_classes=1000):
    config = CONFIGS['ViT_S_16_224_1k']
    config.transformer.dropout_rate = 0.0
    model = VisionTransformer(config,img_size=image_size, n_classes=n_classes)
    if pretrained:
        try:
            model.load_from(np.load(Pretrained['ViT_S_16_224_1k']))
            print("Load pre-trained ViT"+Pretrained['ViT_S_16_224_1k'])
        except:
            print("Unable to load pre-trained model, please put the file in " + Pretrained['ViT_S_16_224_1k'])
    return model

def vit_s_16_224_1k_lora(pretrained=True, image_size=224,  n_classes=1000,  r = 4,alpha=4,lora_layer=None, vis=False):
    config = CONFIGS['ViT_S_16_224_1k']
    config.transformer.dropout_rate = 0.0
    model = VisionTransformer(config,img_size=image_size, n_classes=n_classes, vis=vis)
    if pretrained:
        try:
            model.load_from(np.load(Pretrained['ViT_S_16_224_1k']))
            print("Load pre-trained ViT"+Pretrained['ViT_S_16_224_1k'])
        except:
            print("Unable to load pre-trained model, please put the file in " + Pretrained['ViT_S_16_224_1k'])
    # lets freeze first
    model = LoRA_ViT(model, r, alpha, lora_layer)
    return model



if __name__ == '__main__':
    model = vit_b_16_224(pretrained=False)
    from torchsummary import summary
    import timm
    for name, param in model.named_parameters():
        print(name)
    model = timm.create_model('vit_small_patch16_224',pretrained=False).cuda()
    summary(model, (3, 224, 224))
