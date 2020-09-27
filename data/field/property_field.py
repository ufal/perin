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


class PropertyField(RawField):
    def __init__(self, preprocessing):
        super(PropertyField, self).__init__(preprocessing=preprocessing)

    def process(self, example, device=None):
        example = self.numericalize(example)
        tensor = self.pad(example, device)

        return tensor

    def pad(self, example, device):
        tensor = torch.tensor(example, dtype=torch.long, device=device)
        return tensor

    def numericalize(self, arr):
        def multi_stoi(array):
            if isinstance(array, list):
                return [multi_stoi(a) for a in array]

            output = []
            for key in self.keys:
                if isinstance(array[key], int):
                    output.append(array[key])
                elif array[key] not in self.vocabs[key].stoi:
                    output.append(0)
                else:
                    output.append(self.vocabs[key].stoi[array[key]])
            return output

        return multi_stoi(arr)

    def build_vocab(self, *args):
        def generate(l):
            if isinstance(l, dict):
                yield l
            elif isinstance(l, list) or isinstance(l, types.GeneratorType):
                for i in l:
                    yield from generate(i)
            else:
                raise Exception()

        sources = []
        for arg in args:
            if isinstance(arg, torch.utils.data.Dataset):
                sources += [arg.get_examples(name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        first = True
        for d in generate(sources):
            if d is not None:
                if first:
                    self.keys = sorted(d.keys())
                    self.n_properties = len(self.keys)
                    counters = {key: Counter() for key, _ in d.items()}
                    first = False
                for key, value in d.items():
                    counters[key].update([value])

        self.vocabs = {key: Vocab(counter, specials=[]) for key, counter in counters.items()}
