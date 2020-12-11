#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pickle

import torch
import torchtext
from random import Random

from data.parser.from_mrp.amr_parser import AMRParser
from data.parser.from_mrp.drg_parser import DRGParser
from data.parser.from_mrp.eds_parser import EDSParser
from data.parser.from_mrp.ptg_parser import PTGParser
from data.parser.from_mrp.ucca_parser import UCCAParser
from data.parser.from_mrp.evaluation_parser import EvaluationParser
from data.parser.from_mrp.request_parser import RequestParser
from data.field.edge_field import EdgeField
from data.field.edge_label_field import EdgeLabelField
from data.field.edge_permutation_field import EdgePermutationField
from data.field.field import Field
from data.field.nested_field import NestedField
from data.field.basic_field import BasicField
from data.field.bert_field import BertField
from data.field.relative_label_field import RelativeLabelField
from data.field.property_field import PropertyField
from data.field.anchor_field import AnchorField
from data.batch import Batch
from utility.label_processor import LabelProcessor


def char_tokenize(word):
    return [c for i, c in enumerate(word) if i < 10 or len(word) - i <= 10]


class Collate:
    def __call__(self, batch):
        batch.sort(key=lambda example: example["every_input"][0].size(0), reverse=True)
        return Batch.build(batch)


class Dataset:
    def __init__(self, args):
        self.sos, self.eos, self.pad, self.unk = "<sos>", "<eos>", "<pad>", "<unk>"
        self.processor = LabelProcessor()

        self.bert_input_field = BertField()
        self.scatter_field = BasicField()
        self.every_word_input_field = Field(lower=True, init_token=self.sos, eos_token=self.eos, batch_first=True, include_lengths=True)
        self.every_lemma_field = Field(lower=True, init_token=self.sos, eos_token=self.eos, batch_first=True)

        char_form_nesting = torchtext.data.Field(tokenize=char_tokenize, init_token=self.sos, eos_token=self.eos, batch_first=True)
        self.char_form_field = NestedField(char_form_nesting, include_lengths=True)

        char_lemma_nesting = torchtext.data.Field(tokenize=char_tokenize, init_token=self.sos, eos_token=self.eos, batch_first=True)
        self.char_lemma_field = NestedField(char_lemma_nesting, include_lengths=True)

        self.relative_label_field = RelativeLabelField(
            args.label_smoothing, preprocessing=lambda nodes: [n["possible rules"] for n in nodes],
        )
        self.label_field = Field(preprocessing=lambda nodes: [n["label"].lower() for n in nodes], include_lengths=True, batch_first=True,)
        self.property_field = PropertyField(preprocessing=lambda nodes: [n["properties"] for n in nodes])

        self.id_field = Field(batch_first=True)
        self.edge_presence_field = EdgeField()
        self.edge_attribute_field = EdgeField()
        self.edge_label_field = EdgeLabelField()
        self.edge_permutation_field = EdgePermutationField()
        self.anchor_field = AnchorField()
        self.token_interval_field = BasicField()
        self.top_field = BasicField()

    def load_state_dict(self, args, d):
        self.property_keys = d["property keys"]
        self.property_field.vocabs = pickle.loads(d["property vocabs"])
        for key, value in d["vocabs"].items():
            getattr(self, key).vocab = pickle.loads(value)

    def state_dict(self):
        return {
            "property vocabs": pickle.dumps(self.property_field.vocabs),
            "property keys": self.property_keys,
            "vocabs": {key: pickle.dumps(value.vocab) for key, value in self.__dict__.items() if hasattr(value, "vocab")}
        }

    def load_sentences(self, sentences, args, language: str):
        dataset = RequestParser(
            sentences, args, language,
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "lemmas": [("every_lemma", self.every_lemma_field), ("char_lemma_input", self.char_lemma_field)],
                "to scatter": ("input_scatter", self.scatter_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
        )

        self.every_word_input_field.build_vocab(dataset, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.every_lemma_field.build_vocab(dataset, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.id_field.build_vocab(dataset, min_freq=1, specials=[])

        return dataset

    def load_dataset(self, args, gpu, n_gpus, framework: str, language: str):
        dataset = {
            ("amr", "eng"): AMRParser, ("amr", "zho"): EDSParser,
            ("drg", "eng"): DRGParser, ("drg", "deu"): DRGParser,
            ("eds", "eng"): EDSParser,
            ("ptg", "eng"): PTGParser, ("ptg", "ces"): PTGParser,
            ("ucca", "eng"): UCCAParser, ("ucca", "deu"): UCCAParser,
        }[(framework, language)]

        self.train = dataset(
            args, framework, language, "training",
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "lemmas": ("char_lemma_input", self.char_lemma_field),
                "bert input": ("input", self.bert_input_field),
                "to scatter": ("input_scatter", self.scatter_field),
                "edge permutations": ("edge_permutations", self.edge_permutation_field),
                "nodes": [
                    ("labels", self.label_field),
                    ("relative_labels", self.relative_label_field),
                    ("properties", self.property_field),
                ],
                "top": ("top", self.top_field),
                "edge presence": ("edge_presence", self.edge_presence_field),
                "edge labels": ("edge_labels", self.edge_label_field),
                "edge attributes": ("edge_attributes", self.edge_attribute_field),
                "anchor edges": ("anchor", self.anchor_field),
            },
            filter_pred=lambda example: len(example.input) <= 80,
        )

        self.val = dataset(
            args, framework, language, "validation",
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "lemmas": [("every_lemma", self.every_lemma_field), ("char_lemma_input", self.char_lemma_field)],
                "to scatter": ("input_scatter", self.scatter_field),
                "edge permutations": ("edge_permutations", self.edge_permutation_field),
                "nodes": [
                    ("labels", self.label_field),
                    ("relative_labels", self.relative_label_field),
                    ("properties", self.property_field),
                ],
                "top": ("top", self.top_field),
                "edge presence": ("edge_presence", self.edge_presence_field),
                "edge labels": ("edge_labels", self.edge_label_field),
                "edge attributes": ("edge_attributes", self.edge_attribute_field),
                "anchor edges": ("anchor", self.anchor_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
            precomputed_dataset=self.train.data,
        )

        self.test = EvaluationParser(
            args, framework, language,
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "lemmas": [("every_lemma", self.every_lemma_field), ("char_lemma_input", self.char_lemma_field)],
                "to scatter": ("input_scatter", self.scatter_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
        )

        del self.train.data, self.val.data, self.test.data
        for f in list(self.train.fields.values()) + list(self.val.fields.values()) + list(self.test.fields.values()):
            if hasattr(f, "preprocessing"):
                del f.preprocessing

        self.train_size = len(self.train)
        self.val_size = len(self.val)
        self.test_size = len(self.test)

        print(f"\n{self.train_size} sentences in the train split")
        print(f"{self.val_size} sentences in the validation split")
        print(f"{self.test_size} sentences in the test split")

        self.node_count = self.train.node_counter
        self.token_count = self.train.input_count
        self.edge_count = self.train.edge_counter
        self.no_edge_count = self.train.no_edge_counter
        self.anchor_freq = self.train.anchor_freq
        print(f"{self.node_count} nodes in the train split")

        self.every_word_input_field.build_vocab(self.val, self.test, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.char_form_field.build_vocab(self.train, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.char_lemma_field.build_vocab(self.train, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.every_lemma_field.build_vocab(self.val, self.test, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.label_field.build_vocab(self.train, min_freq=1, specials=[])
        self.property_field.build_vocab(self.train)
        self.relative_label_field.build_vocab(self.train)
        self.id_field.build_vocab(self.val, self.test, min_freq=1, specials=[])
        self.edge_label_field.build_vocab(self.train)
        self.edge_attribute_field.build_vocab(self.train)

        self.create_label_freqs(args)
        self.create_edge_freqs(args)
        self.create_property_freqs(args)
        self.create_top_freqs(args)

        self.property_keys = self.property_field.keys
        print("properties: ", self.property_field.keys)

        print(f"Edge frequency: {self.edge_presence_freq*100:.2f} %")
        print(f"{len(self.relative_label_field.vocab)} words in the relative label vocabulary")
        print(f"{len(self.edge_label_field.vocab)} words in the edge label vocabulary")
        print(f"{len(self.char_form_field.vocab)} characters in the vocabulary")

        Random(42).shuffle(self.train.examples)
        self.train.examples = self.train.examples[:len(self.train.examples) // n_gpus * n_gpus]
        self.train.examples = self.train.examples[gpu * len(self.train.examples) // n_gpus: (gpu + 1) * len(self.train.examples) // n_gpus]

    def relative_output_tensor_to_str(self, relative_label, anchors, tokens, lemmas, concat_rules: bool, num_lemmas=False, ignore_nonalnum=False):
        relative_label_str = self.relative_label_field.vocab.itos[relative_label]

        input_strings = [self.every_word_input_field.vocab.itos[i.item()] for i in tokens[anchors]]
        lemma_strings = [self.every_lemma_field.vocab.itos[i.item()] for i in lemmas[anchors]]

        absolute_label = self.processor.apply_label_rule(input_strings, lemma_strings, relative_label_str, concat_rules, num_lemmas, ignore_nonalnum)

        if absolute_label is None:
            absolute_label = "<none>"
        return absolute_label

    def create_label_freqs(self, args):
        n_rules = len(self.relative_label_field.vocab)
        blank_count = (args.query_length * self.token_count - self.node_count) * args.blank_weight
        blank_p = blank_count * (1.0 - args.label_smoothing) + self.node_count * args.label_smoothing / n_rules
        non_blank_p = blank_count * args.label_smoothing / n_rules
        label_counts = [blank_p] + [
            self.train.rule_counter[self.relative_label_field.vocab.itos[i]] + non_blank_p
            for i in range(n_rules)
        ]
        label_counts = torch.FloatTensor(label_counts)
        self.label_freqs = label_counts / (self.node_count + blank_count)

    def create_edge_freqs(self, args):
        edge_counter = [
            self.edge_label_field.vocab.freqs[self.edge_label_field.vocab.itos[i]] for i in range(len(self.edge_label_field.vocab))
        ]
        edge_counter = torch.FloatTensor(edge_counter)
        self.edge_label_freqs = edge_counter / self.edge_count
        self.edge_presence_freq = self.edge_count / (self.edge_count + self.no_edge_count)
        edge_attribute_counter = [
            self.edge_attribute_field.vocab.freqs[self.edge_attribute_field.vocab.itos[i]] for i in range(len(self.edge_attribute_field.vocab))
        ]
        edge_attribute_counter = torch.FloatTensor(edge_attribute_counter)
        self.edge_attribute_freqs = edge_attribute_counter / self.edge_count

    def create_property_freqs(self, args):
        property_counter = {
            key: [vocab.freqs[vocab.itos[i]] for i in range(len(vocab))] for key, vocab in self.property_field.vocabs.items()
        }
        self.property_freqs = {key: torch.FloatTensor(c) / self.node_count for key, c in property_counter.items()}

    def create_top_freqs(self, args):
        self.top_freq = self.train_size / self.node_count
