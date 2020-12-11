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


class PaddingPacker(nn.Module):
    def __init__(self, module):
        super(PaddingPacker, self).__init__()
        self.module = module

    def forward(self, x, lengths=None, total_length=None):
        if lengths is not None:
            package = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
            x = package.data

        x = self.module(x)

        if lengths is not None:
            x = self.unpack(package, x, total_length=total_length)

        return x

    def unpack(self, original_package, data, total_length=None):
        new_package = nn.utils.rnn.PackedSequence(
            data, batch_sizes=original_package.batch_sizes, sorted_indices=original_package.sorted_indices
        )
        return nn.utils.rnn.pad_packed_sequence(new_package, batch_first=True, total_length=total_length)[0]
