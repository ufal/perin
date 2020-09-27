#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformers.modeling_bert import BertModel
from model.transformers.modeling_roberta import RobertaModel
from model.module.char_embedding import CharEmbedding


class WordDropout(nn.Dropout):
    def forward(self, input_tensor):
        if self.p == 0:
            return input_tensor

        ones = input_tensor.new_ones(input_tensor.shape[:-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)

        return dropout_mask.unsqueeze(-1) * input_tensor


class QueryGenerator(nn.Module):
    def __init__(self, dim, width_factor, n_frameworks):
        super(QueryGenerator, self).__init__()

        weight = torch.Tensor(width_factor * dim, dim)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight.t().repeat(n_frameworks, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_frameworks, width_factor * dim))
        self.width_factor = width_factor

    def forward(self, encoder_output, frameworks):
        batch_size, seq_len, dim = encoder_output.shape
        weight = self.weight[frameworks, :, :]  # shape: (B, D, Q*D)
        bias = self.bias[frameworks, :].unsqueeze(1)  # shape: (B, 1, Q*D)

        queries = encoder_output.matmul(weight) + bias  # shape: (B, T, Q*D)
        queries = torch.tanh(queries)  # shape: (B, T, Q*D)
        queries = queries.view(batch_size, seq_len, self.width_factor, dim).flatten(1, 2)  # shape: (B, T*Q, D)
        return queries


class Encoder(nn.Module):
    def __init__(self, args, dataset):
        super(Encoder, self).__init__()

        self.dim = args.hidden_size
        self.n_layers = args.n_encoder_layers
        self.width_factor = args.query_length

        if "roberta" in args.encoder.lower():
            self.bert = RobertaModel.from_pretrained(args.encoder, output_hidden_states=True)
            if args.encoder_freeze_embedding:
                self.bert.embeddings.requires_grad_(False)
        else:
            self.bert = BertModel.from_pretrained(args.encoder, output_hidden_states=True)

        self.bert.pooler = nn.Identity()  # effectively delete the pooler (for DistributedDataParallel)

        self.use_char_embedding = args.char_embedding
        if self.use_char_embedding:
            self.form_char_embedding = CharEmbedding(dataset.char_form_vocab_size, args.char_embedding_size, self.dim)
            self.lemma_char_embedding = CharEmbedding(dataset.char_lemma_vocab_size, args.char_embedding_size, self.dim)
            self.word_dropout = WordDropout(args.dropout_word)

        self.query_generator = QueryGenerator(self.dim, self.width_factor, len(args.frameworks))
        self.encoded_layer_norm = nn.LayerNorm(self.dim)
        self.scores = nn.Parameter(torch.zeros(self.n_layers, 1, 1, 1), requires_grad=True)

    def forward(self, bert_input, form_chars, lemma_chars, to_scatter, n_words, frameworks):
        tokens, mask = bert_input
        batch_size = tokens.size(0)

        encoded = self.bert(tokens, attention_mask=mask)[2][1:]
        encoded = torch.stack(encoded, dim=0)  # shape: (12, B, T, H)
        encoded = self.encoded_layer_norm(encoded)

        if self.training:
            time_len = encoded.size(2)
            scores = self.scores.expand(-1, batch_size, time_len, -1)
            dropout = torch.empty(self.n_layers, batch_size, 1, 1, dtype=torch.bool, device=self.scores.device)
            dropout.bernoulli_(0.1)
            scores = scores.masked_fill(dropout, float("-inf"))
        else:
            scores = self.scores

        scores = F.softmax(scores, dim=0)
        encoded = (scores * encoded).sum(0)  # shape: (B, T, H)

        to_scatter = to_scatter.unsqueeze(-1).expand(-1, -1, self.dim)
        encoder_output = torch.zeros(encoded.size(0), n_words + 1, self.dim, device=encoded.device)
        encoder_output.scatter_add_(dim=1, index=to_scatter, src=encoded[:, 1:-1, :])  # shape: (B, n_words + 1, H)
        encoder_output = encoder_output[:, :-1, :]

        decoder_input = self.query_generator(encoder_output, frameworks)

        if self.use_char_embedding:
            form_char_embedding = self.form_char_embedding(form_chars[0], form_chars[1], form_chars[2])
            lemma_char_embedding = self.lemma_char_embedding(lemma_chars[0], lemma_chars[1], lemma_chars[2])
            encoder_output = self.word_dropout(encoder_output) + form_char_embedding + lemma_char_embedding

        return encoder_output, decoder_input
