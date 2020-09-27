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
import torch.nn.functional as F


class MixtureOfSoftmaxes(nn.Module):
    def __init__(self, dataset, args, initialize: bool):
        super(MixtureOfSoftmaxes, self).__init__()

        self.n_heads = args.n_mixture_components
        self.out_dim = len(dataset.relative_label_field.vocab) + 1
        self.hidden_size = args.hidden_size

        self.prior = nn.Linear(self.hidden_size, self.n_heads)
        self.latent = nn.Linear(self.hidden_size, self.n_heads * self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.out_dim)

        if initialize:
            self.output.bias.data = dataset.label_freqs.log()

    def forward(self, hidden):
        prior = self.prior(hidden).sigmoid()  # shape: (B, T, N)
        prior = prior / (prior.sum(-1, keepdim=True) + 1e-8)

        latent = self.latent(hidden).view(*(hidden.shape[:-1] + (self.n_heads, self.hidden_size))).tanh()  # shape: (B, T, N, H)
        output = F.softmax(self.output(latent), dim=-1)  # shape: (B, T, N, H)

        output = (prior.unsqueeze(-1) * output).sum(-2)  # shape: (B, T, H)
        return output
