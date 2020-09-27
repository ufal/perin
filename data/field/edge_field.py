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
from torchtext.vocab import Vocab
from collections import Counter
import types


class EdgeField(RawField):
    def __init__(self):
        super(EdgeField, self).__init__()
        self.vocab = None

    def process(self, edges, device=None):
        edges = self.numericalize(edges)
        tensor = self.pad(edges, device)
        return tensor

    def pad(self, edges, device):
        tensor = torch.zeros(edges[0], edges[1], dtype=torch.long, device=device)
        for edge in edges[-1]:
            tensor[edge[0], edge[1]] = edge[2]

        return tensor

    def numericalize(self, arr):
        def multi_map(array, function):
            if isinstance(array, tuple):
                return (array[0], array[1], function(array[2]))
            elif isinstance(array, list):
                return [multi_map(array[i], function) for i in range(len(array))]
            else:
                return array

        if self.vocab is not None:
            arr = multi_map(arr, lambda x: self.vocab.stoi[x] if x is not None else 0)
        return arr

    def build_vocab(self, *args):
        def generate(l):
            if isinstance(l, tuple):
                yield l[2]
            elif isinstance(l, list) or isinstance(l, types.GeneratorType):
                for i in l:
                    yield from generate(i)
            else:
                return

        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, torch.utils.data.Dataset):
                sources += [arg.get_examples(name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for x in generate(sources):
            if x is not None:
                counter.update([x])

        self.vocab = Vocab(counter, specials=[])
