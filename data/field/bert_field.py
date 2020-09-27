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


class BertField(RawField):
    def __init__(self):
        super(BertField, self).__init__()

    def process(self, example, device=None):
        attention_mask = [1] * len(example)

        example = torch.LongTensor(example, device=device)
        attention_mask = torch.ones_like(example)

        return example, attention_mask
