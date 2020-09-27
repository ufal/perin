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


class EdgePermutationField(RawField):
    def __init__(self):
        super(EdgePermutationField, self).__init__()

    def process(self, example, device=None):
        permutations = torch.LongTensor(example["permutations"], device=device)
        masks = self.generate_mask(len(example["permutations"][0]), example["greedy"], device)
        greedies = [torch.LongTensor(p, device=device) for p in example["greedy"]]

        return permutations, masks, greedies

    def generate_mask(self, length, greedy, device):
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        for g in greedy:
            mask[g] = True
        return mask
