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


def scale_grad(x, weight):
    return weight * x + ((1.0 - weight) * x).detach()


class GradScaler(nn.Module):
    def __init__(self, weight):
        super(GradScaler, self).__init__()
        self.weight = (weight,)  # wrap the weight as a "pointer" to not add it as parameter

    def forward(self, x):
        return scale_grad(x, self.weight[0])
