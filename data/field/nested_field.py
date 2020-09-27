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
import torchtext


class NestedField(torchtext.data.NestedField):
    def pad(self, example):
        self.nesting_field.include_lengths = self.include_lengths
        if not self.include_lengths:
            return self.nesting_field.pad(example)

        sentence_length = len(example)
        example, word_lengths = self.nesting_field.pad(example)
        return example, sentence_length, word_lengths

    def numericalize(self, arr, device=None):
        numericalized = []
        self.nesting_field.include_lengths = False
        if self.include_lengths:
            arr, sentence_length, word_lengths = arr

        numericalized = self.nesting_field.numericalize(arr, device=device)

        self.nesting_field.include_lengths = True
        if self.include_lengths:
            sentence_length = torch.tensor(sentence_length, dtype=self.dtype, device=device)
            word_lengths = torch.tensor(word_lengths, dtype=self.dtype, device=device)
            return (numericalized, sentence_length, word_lengths)
        return numericalized

    def build_vocab(self, *args, **kwargs):
        sources = []
        for arg in args:
            if isinstance(arg, torch.utils.data.Dataset):
                sources += [arg.get_examples(name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        flattened = []
        for source in sources:
            flattened.extend(source)

        # just build vocab and does not load vector
        self.nesting_field.build_vocab(*flattened, **kwargs)
        super(torchtext.data.NestedField, self).build_vocab()
        self.vocab.extend(self.nesting_field.vocab)
        self.vocab.freqs = self.nesting_field.vocab.freqs.copy()
        self.nesting_field.vocab = self.vocab
