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
from torchtext.data import RawField


class AnchorField(RawField):
    def process(self, batch, device=None):
        tensors, masks = self.pad(batch, device)
        return tensors, masks

    def pad(self, anchors, device):
        tensor = torch.zeros(anchors[0], anchors[1], dtype=torch.long, device=device)
        for anchor in anchors[-1]:
            tensor[anchor[0], anchor[1]] = 1
        mask = tensor.sum(-1) == 0

        return tensor, mask
