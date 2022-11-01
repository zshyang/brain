import copy
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList, T
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
import torch.nn.functional as F
from torch import Tensor

# from polygon import *


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """ Randomly drop some values along the first dimension of the input tensor
    given the drop probability.
    This function is used for class DropPath.

    Args:
        x: The input tensor. It could be either on GPU or not.
        drop_prob: The probability of dropping.
        training: Whether this is happened during training or not. If it is not
            during training, then no drop should be applied.

    Returns:
        output: The output tensor.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1,...)
    random_tensor = keep_prob + torch.rand(shape,
                                           dtype=x.dtype, 
                                           device=x.device)
    random_tensor.floor_()  # binarize
    # I guess the division here is a way to keep the energy equal.
    # The reference about torch division:
    # https://pytorch.org/docs/stable/generated/torch.div.html
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of 
    residual blocks).
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """ A self attention layer.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """ The forward function. And the input and output would have the same 
        shape.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    """ A block of transformer.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

""" The following classes are copied from 
https://github.com/pytorch/pytorch/blob/cab926b2c027739e322d922e5d8df5479d40ad4c/torch/nn/modules/transformer.py
"""

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class 
            (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch 
                (optional).
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, 
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You 
    Need". Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion 
    Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention 
    is all you need. In Advances in Neural Information Processing Systems, pages
    6000-6010. Users may modify or implement in a different way during 
    application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model 
            (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu 
            (default=relu).
        layer_norm_eps: the eps value in layer normalization components 
            (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", layer_norm_eps=1e-5, batch_first=False, 
                 norm_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.batch_first = batch_first
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch 
                (optional).
        """
        if self.batch_first:
            src = src.transpose(0, 1)

        if self.norm_first:
            src = self.norm1(src)
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(
                self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(
                self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        if self.batch_first:
            src = src.transpose(0, 1)

        return src

class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class 
            (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, 
                num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch 
                (optional).
            memory_key_padding_mask: the mask for the memory keys per batch 
                (optional).
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and 
    feedforward network. This standard decoder layer is based on the paper 
    "Attention Is All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar, 
    Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia 
    Polosukhin. 2017. Attention is all you need. In Advances in Neural 
    Information Processing Systems, pages 6000-6010. Users may modify or 
    implement in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model 
            (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu 
            (default=relu).
        layer_norm_eps: the eps value in layer normalization components 
            (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to self attention, 
            multihead attention and feedforward operations, respectivaly. 
            Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, 
                batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", layer_norm_eps=1e-5, batch_first=False, 
                 norm_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, 
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, 
                                                 dropout=dropout,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.batch_first = batch_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch 
                (optional).
            memory_key_padding_mask: the mask for the memory keys per batch 
                (optional).
        """
        if self.batch_first:
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)

        if self.norm_first:
            tgt = self.norm1(tgt)
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                tgt, memory, memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm3(tgt)
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
        else:  # norm last
            tgt2 = self.self_attn(
                tgt, tgt, tgt, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(
                tgt, memory, memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        if self.batch_first:
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)

        return tgt

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. 
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask

def test():
    pass

if __name__ == "__main__":
    test()
