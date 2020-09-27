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
from model.module.bilinear import Bilinear


class Biaffine(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, bias_init=None):
        super(Biaffine, self).__init__()

        self.linear_1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_2 = nn.Linear(input_dim, output_dim, bias=False)

        self.bilinear = Bilinear(input_dim, input_dim, output_dim, bias=bias)
        if bias_init is not None:
            self.bilinear.bias.data = bias_init

    def forward(self, x, y):
        return self.bilinear(x, y) + self.linear_1(x).unsqueeze(2) + self.linear_2(y).unsqueeze(1)
