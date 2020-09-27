#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import Counter

import torch
from torchtext.vocab import Vocab

from data.dataset import Dataset, Collate
from data.batch import Batch
from data.concat_dataset import ConcatDataset


class SharedDataset:
    def __init__(self, args):
        self.child_datasets = {
            (framework, language): Dataset(args) for framework, language in args.frameworks
        }
        self.framework_to_id = {(f, l): i for i, (f, l) in enumerate(args.frameworks)}
        self.id_to_framework = {i: (f, l) for i, (f, l) in enumerate(args.frameworks)}

    def load_state_dict(self, args, d):
        for key, dataset in self.child_datasets.items():
            dataset.load_state_dict(args, d[key])
        self.share_chars()

    def state_dict(self):
        return {key: dataset.state_dict() for key, dataset in self.child_datasets.items()}

    def load_sentences(self, sentences, args, framework: str, language: str):
        def switch(f, l, s):
            return s if (framework == f and language == l) else []

        datasets = [
            dataset.load_sentences(switch(f, l, sentences), args, language)
            for (f, l), dataset in self.child_datasets.items()
        ]
        return torch.utils.data.DataLoader(ConcatDataset(datasets), batch_size=1, shuffle=False, collate_fn=Collate())

    def load_datasets(self, args, gpu, n_gpus):
        for (framework, language), dataset in self.child_datasets.items():
            dataset.load_dataset(args, gpu, n_gpus, framework, language)

        self.share_chars()
        self.share_vocabs(args)

        train_datasets = [self.child_datasets[self.id_to_framework[i]].train for i in range(len(self.child_datasets))]
        self.train = torch.utils.data.DataLoader(
            ConcatDataset(train_datasets),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=Collate(),
            pin_memory=True,
            drop_last=True
        )
        self.train_size = len(self.train.dataset)
        self.mean_label_length = sum(dataset.node_count for dataset in self.child_datasets.values()) / self.train_size

        val_datasets = [self.child_datasets[self.id_to_framework[i]].val for i in range(len(self.child_datasets))]
        self.val = torch.utils.data.DataLoader(
            ConcatDataset(val_datasets),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=Collate(),
            pin_memory=True,
        )
        self.val_size = len(self.val.dataset)

        test_datasets = [self.child_datasets[self.id_to_framework[i]].test for i in range(len(self.child_datasets))]
        self.test = torch.utils.data.DataLoader(
            ConcatDataset(test_datasets),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=Collate(),
            pin_memory=True,
        )
        self.test_size = len(self.test.dataset)

        if gpu == 0:
            batch = next(iter(self.train))
            print(f"\nBatch content: {Batch.to_str(batch)}\n")
            print(flush=True)

    def share_chars(self):
        sos, eos, unk, pad = "<sos>", "<eos>", "<unk>", "<pad>"

        form_counter, lemma_counter = Counter(), Counter()
        for dataset in self.child_datasets.values():
            form_counter += dataset.char_form_field.vocab.freqs
            lemma_counter += dataset.char_lemma_field.vocab.freqs

        form_vocab = Vocab(form_counter, min_freq=1, specials=[pad, unk, sos, eos])
        lemma_vocab = Vocab(lemma_counter, min_freq=1, specials=[pad, unk, sos, eos])

        for dataset in self.child_datasets.values():
            dataset.char_form_field.vocab = dataset.char_form_field.nesting_field.vocab = form_vocab
            dataset.char_lemma_field.vocab = dataset.char_lemma_field.nesting_field.vocab = lemma_vocab

        self.char_form_vocab_size = len(form_vocab)
        self.char_lemma_vocab_size = len(lemma_vocab)

    def share_vocabs(self, args):
        ucca_datasets = [dataset for (f, l), dataset in self.child_datasets.items() if f == "ucca"]
        if len(ucca_datasets) == 2:
            print("sharing UCCA vocabs...")
            self.share_vocabs_(ucca_datasets[0], ucca_datasets[1], args, share_edges=True, share_anchors=True, share_labels=True)

        ptg_datasets = [dataset for (f, l), dataset in self.child_datasets.items() if f == "ptg"]
        if len(ptg_datasets) == 2:
            print("sharing PTG vocabs...")
            self.share_vocabs_(ptg_datasets[0], ptg_datasets[1], args, share_edges=True, share_anchors=True)

        drg_datasets = [dataset for (f, l), dataset in self.child_datasets.items() if f == "drg"]
        if len(drg_datasets) == 2:
            print("sharing DRG vocabs...")
            self.share_vocabs_(drg_datasets[0], drg_datasets[1], args, share_edges=True, share_tops=True, share_properties=True)

        amr_datasets = [dataset for (f, l), dataset in self.child_datasets.items() if f == "amr"]
        if len(amr_datasets) == 2:
            print("sharing AMR vocabs...")
            self.share_vocabs_(amr_datasets[0], amr_datasets[1], args, share_edges=True, share_tops=True, share_properties=True)

    def share_vocabs_(self, a, b, args, share_edges=False, share_anchors=False, share_labels=False, share_tops=False, share_properties=False):
        a.node_count = b.node_count = a.node_count + b.node_count
        a.token_count = b.token_count = a.token_count + b.token_count

        if share_edges:
            a.edge_count = b.edge_count = a.edge_count + b.edge_count
            a.no_edge_count = b.no_edge_count = a.no_edge_count + b.no_edge_count

            edge_label_counter = a.edge_label_field.vocab.freqs + b.edge_label_field.vocab.freqs
            a.edge_label_field.vocab = b.edge_label_field.vocab = Vocab(edge_label_counter, specials=[])

            edge_attribute_counter = a.edge_attribute_field.vocab.freqs + b.edge_attribute_field.vocab.freqs
            a.edge_attribute_field.vocab = b.edge_attribute_field.vocab = Vocab(edge_attribute_counter, specials=[])

            a.create_edge_freqs(args)
            b.create_edge_freqs(args)

        if share_anchors:
            a.anchor_freq = b.anchor_freq = (a.train_size * a.anchor_freq + b.train_size * b.anchor_freq) / (a.train_size + b.train_size)

        if share_tops:
            a.train_size = b.train_size = a.train_size + b.train_size
            a.create_top_freqs(args)
            b.create_top_freqs(args)

        if share_labels:
            label_counter = a.relative_label_field.vocab.freqs + b.relative_label_field.vocab.freqs
            a.relative_label_field.vocab = b.relative_label_field.vocab = Vocab(label_counter, specials=[])
            a.train.rule_counter = b.train.rule_counter = a.train.rule_counter + b.train.rule_counter
            a.create_label_freqs(args)
            b.create_label_freqs(args)

        if share_properties:
            for key in a.property_field.vocabs.keys():
                assert key in b.property_field.vocabs
                property_counter = a.property_field.vocabs[key].freqs + b.property_field.vocabs[key].freqs
                a.property_field.vocabs[key] = b.property_field.vocabs[key] = Vocab(property_counter, specials=[])
            a.create_property_freqs(args)
            b.create_property_freqs(args)
