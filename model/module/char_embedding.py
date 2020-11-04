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
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class CharEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, output_size: int):
        super(CharEmbedding, self).__init__()

        self.hidden_size = output_size // 2

        self.embedding = nn.Embedding(vocab_size, embedding_size, sparse=False)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.gru = nn.GRU(embedding_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.layer_norm_2 = nn.LayerNorm(2 * self.hidden_size)

    def forward(self, words, sentence_lens, word_lens):
        # input shape: (B, W, C)
        n_words = words.size(1)
        sentence_lens = sentence_lens.cpu()
        sentence_packed = pack_padded_sequence(words, sentence_lens, batch_first=True)  # shape: (B*W, C)
        lens_packed = pack_padded_sequence(word_lens, sentence_lens, batch_first=True)  # shape: (B*W)
        word_packed = pack_padded_sequence(sentence_packed.data, lens_packed.data.cpu(), batch_first=True, enforce_sorted=False)  # shape: (B*W*C)

        embedded = self.embedding(word_packed.data)  # shape: (B*W*C, D)
        embedded = self.layer_norm(embedded)  # shape: (B*W*C, D)

        embedded_packed = PackedSequence(embedded, word_packed[1], word_packed[2], word_packed[3])
        _, embedded = self.gru(embedded_packed)  # shape: (layers * 2, B*W, D)

        embedded = embedded[-2:, :, :].transpose(0, 1).flatten(1, 2)  # shape: (B*W, 2*D)
        embedded, _ = pad_packed_sequence(
            PackedSequence(embedded, sentence_packed[1], sentence_packed[2], sentence_packed[3]), batch_first=True, total_length=n_words,
        )  # shape: (B, W, 2*D)
        embedded = self.layer_norm_2(embedded)

        return embedded  # shape: (B, W, 2*D)
