#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.biaffine import Biaffine
from model.module.grad_scaler import scale_grad


class EdgeClassifier(nn.Module):
    def __init__(self, dataset, args, initialize: bool, presence: bool, label: bool, attribute: bool):
        super(EdgeClassifier, self).__init__()

        self.presence = presence
        if self.presence:
            if initialize:
                presence_init = torch.tensor([dataset.edge_presence_freq])
                presence_init = (presence_init / (1.0 - presence_init)).log()
            else:
                presence_init = None

            self.edge_presence = EdgeBiaffine(
                args.hidden_size, args.hidden_size_edge_presence, 1, args.dropout_edge_label, bias_init=presence_init
            )

        self.label = label
        if self.label:
            label_init = (dataset.edge_label_freqs / (1.0 - dataset.edge_label_freqs)).log() if initialize else None
            n_labels = len(dataset.edge_label_field.vocab)
            self.edge_label = EdgeBiaffine(
                args.hidden_size, args.hidden_size_edge_label, n_labels, args.dropout_edge_presence, bias_init=label_init
            )

        self.attribute = attribute
        if self.attribute:
            if len(dataset.edge_attribute_field.vocab) == 2:
                n_attributes = 1
                attribute_init = (dataset.edge_attribute_freqs[1:] / (1.0 - dataset.edge_attribute_freqs[1:])).log() if initialize else None
            else:
                n_attributes = len(dataset.edge_attribute_field.vocab)
                attribute_init = dataset.edge_attribute_freqs.log() if initialize else None

            self.edge_attribute = EdgeBiaffine(
                args.hidden_size, args.hidden_size_edge_attribute, n_attributes, args.dropout_edge_attribute, bias_init=attribute_init
            )

    def forward(self, x, loss_weights):
        presence, label, attribute = None, None, None

        if self.presence:
            presence = self.edge_presence(scale_grad(x, loss_weights["edge presence"])).squeeze(-1)  # shape: (B, T, T)
        if self.label:
            label = self.edge_label(scale_grad(x, loss_weights["edge label"]))  # shape: (B, T, T, O_1)
        if self.attribute:
            attribute = self.edge_attribute(scale_grad(x, loss_weights["edge attribute"]))  # shape: (B, T, T, O_2)

        return presence, label, attribute


class EdgeBiaffine(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, output_dim, dropout, bias_init=None):
        super(EdgeBiaffine, self).__init__()
        self.hidden = nn.Linear(hidden_dim, 2 * bottleneck_dim)
        self.output = Biaffine(bottleneck_dim, output_dim, bias_init=bias_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.elu(self.hidden(x)))  # shape: (B, T, 2H)
        predecessors, current = x.chunk(2, dim=-1)  # shape: (B, T, H), (B, T, H)
        edge = self.output(current, predecessors)  # shape: (B, T, T, O)
        return edge
