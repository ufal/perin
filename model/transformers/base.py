#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch.nn as nn
from model.transformers.modeling_bert import checkpoint


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = nn.MultiheadAttention(args.hidden_size, args.n_attention_heads, args.dropout_transformer_attention)
        self.dropout = nn.Dropout(args.dropout_transformer)

    def forward(self, q_input, kv_input, mask=None):
        output, _ = self.attention(q_input, kv_input, kv_input, mask, need_weights=False)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size_ff),
            self._get_activation_f(args.activation),
            nn.Dropout(args.dropout_transformer),
            nn.Linear(args.hidden_size_ff, args.hidden_size),
            nn.Dropout(args.dropout_transformer),
        )

    def forward(self, x):
        return self.f(x)

    def _get_activation_f(self, activation: str):
        return {"relu": nn.ReLU, "gelu": nn.GELU}[activation]()


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_f = Attention(args)
        self.cross_f = Attention(args)
        self.feedforward_f = FeedForward(args)

        self.pre_self_norm = nn.LayerNorm(args.hidden_size) if args.pre_norm else nn.Identity()
        self.pre_cross_norm = nn.LayerNorm(args.hidden_size) if args.pre_norm else nn.Identity()
        self.pre_feedforward_norm = nn.LayerNorm(args.hidden_size) if args.pre_norm else nn.Identity()
        self.post_self_norm = nn.Identity() if args.pre_norm else nn.LayerNorm(args.hidden_size)
        self.post_cross_norm = nn.Identity() if args.pre_norm else nn.LayerNorm(args.hidden_size)
        self.post_feedforward_norm = nn.Identity() if args.pre_norm else nn.LayerNorm(args.hidden_size)

    def forward(self, x, encoder_output, x_mask, encoder_mask):
        x_ = self.pre_self_norm(x)
        x = self.post_self_norm(x + self.self_f(x_, x_, x_mask))

        x_ = self.pre_cross_norm(x)
        x = self.post_cross_norm(x + self.cross_f(x_, encoder_output, encoder_mask))

        x_ = self.pre_feedforward_norm(x)
        x = self.post_feedforward_norm(x + self.feedforward_f(x_))

        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, target, encoder, target_mask, encoder_mask):
        target = target.transpose(0, 1)  # shape: (T, B, D)
        encoder = encoder.transpose(0, 1)  # shape: (T, B, D)

        for layer in self.layers[:-1]:
            target = checkpoint(layer, target, encoder, target_mask, encoder_mask)
        target = self.layers[-1](target, encoder, target_mask, encoder_mask)  # don't checkpoint due to grad_norm
        target = target.transpose(0, 1)  # shape: (B, T, D)

        return target
