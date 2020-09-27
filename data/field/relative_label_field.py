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


class RelativeLabelField(RawField):
    def __init__(self, label_smoothing: float, preprocessing):
        super(RelativeLabelField, self).__init__(preprocessing=preprocessing)
        self.label_smoothing = label_smoothing
        self.vocab = None

    def process(self, example, device=None):
        example = self.numericalize(example)
        tensor, lengths = self.pad(example, device)
        return tensor, lengths

    def pad(self, example, device):
        n_labels = len(self.vocab)
        length = torch.LongTensor([len(example)], device=device)

        n_words, n_tokens = len(example), example[0][0]
        tensor = torch.full([n_words, n_tokens, n_labels + 1], self.label_smoothing / n_labels, dtype=torch.float, device=device)
        for i_word, word in enumerate(example):
            for anchor, count, rule in word[1]:
                tensor[i_word, anchor, rule + 1] = (1.0 - self.label_smoothing) / count

        return tensor, length

    def numericalize(self, arr):
        def multi_map(array, function):
            if isinstance(array, tuple):
                return (array[0], array[1], function(array[2]))
            elif isinstance(array, list):
                return [multi_map(a, function) for a in array]
            else:
                return array

        if self.vocab is not None:
            arr = multi_map(arr, lambda x: self.vocab.stoi[x] if x in self.vocab.stoi else 0)

        return arr

    def build_vocab(self, *args):
        def generate(l):
            if isinstance(l, tuple):
                yield l[2]
            if isinstance(l, list) or isinstance(l, types.GeneratorType):
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
            counter.update([x])

        self.vocab = Vocab(counter, specials=[])
