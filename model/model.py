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
import torch.nn as nn

from model.module.encoder import Encoder

from model.transformers.base import Decoder
from model.head.amr_head import AMRHead
from model.head.drg_head import DRGHead
from model.head.eds_head import EDSHead
from model.head.ptg_head import PTGHead
from model.head.ucca_head import UCCAHead
from model.module.module_wrapper import ModuleWrapper
from utility.utils import create_padding_mask
from data.batch import Batch


class Model(nn.Module):
    def __init__(self, dataset, args, initialize=True):
        super(Model, self).__init__()
        self.encoder = Encoder(args, dataset)
        self.decoder = Decoder(args)

        head_dict = {
            ("amr", "eng"): AMRHead, ("amr", "zho"): EDSHead,
            ("drg", "eng"): DRGHead, ("drg", "deu"): DRGHead,
            ("eds", "eng"): EDSHead,
            ("ptg", "eng"): PTGHead, ("ptg", "ces"): PTGHead,
            ("ucca", "eng"): UCCAHead, ("ucca", "deu"): UCCAHead,
        }

        self.heads = nn.ModuleList([])
        for i in range(len(dataset.child_datasets)):
            f, l = dataset.id_to_framework[i]
            self.heads.append(head_dict[(f, l)](dataset.child_datasets[(f, l)], args, f, l, initialize))

        self.query_length = args.query_length
        self.label_smoothing = args.label_smoothing
        self.total_epochs = args.epochs
        self.dataset = dataset
        self.args = args

        self.share_weights()

    def forward(self, batch, inference=False, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size, input_len = every_input.size(0), every_input.size(1)
        device = every_input.device

        encoder_mask = create_padding_mask(batch_size, input_len, word_lens, device)
        decoder_mask = create_padding_mask(batch_size, self.query_length * input_len, decoder_lens, device)

        encoder_output, decoder_input = self.encoder(
            batch["input"], batch["char_form_input"], batch["char_lemma_input"], batch["input_scatter"], input_len, batch["framework"]
        )

        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, encoder_mask)

        def select_inputs(indices):
            return (
                encoder_output.index_select(0, indices),
                decoder_output.index_select(0, indices),
                encoder_mask.index_select(0, indices),
                decoder_mask.index_select(0, indices),
                Batch.index_select(batch, indices),
            )

        if inference:
            output = {}
            for i, head in enumerate(self.heads):
                indices = (batch["framework"] == i).nonzero(as_tuple=False).flatten()
                if indices.size(0) == 0:
                    continue
                output[self.dataset.id_to_framework[i]] = head.predict(*select_inputs(indices), **kwargs)

            return output

        else:
            total_loss, losses, stats = 0.0, [], {}
            for i, head in enumerate(self.heads):
                indices = (batch["framework"] == i).nonzero(as_tuple=False).flatten()

                if indices.size(0) == 0:
                    args = self.get_dummy_batch(head, device)
                    total_loss_, _, _ = head(*args)
                    total_loss = total_loss + 0.0 * total_loss_
                    losses.append([])
                    continue

                total_loss_, losses_, stats_ = head(*select_inputs(indices))
                lr_mult = torch.cat([batch["relative_labels"][1][j] for j in indices]).float().mean() / self.dataset.mean_label_length
                lr_mult *= indices.size(0) / batch_size / self.args.accumulation_steps
                total_loss = total_loss + total_loss_ * lr_mult
                losses.append(losses_)
                stats.update(stats_)

            return total_loss, losses, stats

    def get_decoder_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith("encoder.bert") and "loss_weights" not in name)

    def get_encoder_parameters(self, n_layers):
        return [
            [p for name, p in self.named_parameters() if name.startswith(f"encoder.bert.encoder.layer.{n_layers - 1 - i}.")] for i in range(n_layers)
        ]

    def share_weights(self):
        ucca_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "ucca"]
        if len(ucca_heads) == 2:
            self.share_weights_(ucca_heads[0], ucca_heads[1], share_labels=True, share_edges=True, share_anchors=True)

        ptg_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "ptg"]
        if len(ptg_heads) == 2:
            self.share_weights_(ptg_heads[0], ptg_heads[1], share_edges=True, share_anchors=True)

        drg_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "drg"]
        if len(drg_heads) == 2:
            self.share_weights_(drg_heads[0], drg_heads[1], share_edges=True, share_tops=True, share_properties=True)

        amr_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "amr"]
        if len(amr_heads) == 2:
            self.share_weights_(amr_heads[0], amr_heads[1], share_edges=True, share_tops=True, share_properties=True)

    def share_weights_(self, a, b, share_edges=False, share_anchors=False, share_labels=False, share_tops=False, share_properties=False):
        if share_edges:
            del b.edge_classifier
            b.edge_classifier = ModuleWrapper(a.edge_classifier)

        if share_anchors:
            del b.anchor_classifier
            b.anchor_classifier = ModuleWrapper(a.anchor_classifier)

        if share_tops:
            del b.top_classifier
            b.top_classifier = ModuleWrapper(a.top_classifier)

        if share_labels:
            del b.label_classifier
            b.label_classifier = ModuleWrapper(a.label_classifier)

        if share_properties:
            del b.property_classifier
            b.property_classifier = ModuleWrapper(a.property_classifier)

    def get_dummy_batch(self, head, device):
        encoder_output = torch.zeros(1, 1, self.args.hidden_size, device=device)
        decoder_output = torch.zeros(1, self.query_length, self.args.hidden_size, device=device)
        encoder_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        decoder_mask = torch.zeros(1, self.query_length, dtype=torch.bool, device=device)
        batch = {
            "every_input": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "input": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "edge_permutations": ([torch.ones(1, 1, dtype=torch.long, device=device)], [torch.zeros(1, dtype=torch.bool, device=device)], [[]]),
            "labels": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "relative_labels": ([torch.zeros(1, 1, len(head.dataset.relative_label_field.vocab) + 1, device=device)], [torch.ones(1, dtype=torch.long, device=device)]),
            "properties": torch.zeros(1, 1, 10, dtype=torch.long, device=device),
            "top": torch.zeros(1, dtype=torch.long, device=device),
            "edge_presence": torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            "edge_labels": (torch.zeros(1, 1, 1, head.dataset.edge_label_freqs.size(0), dtype=torch.long, device=device), torch.zeros(1, 1, 1, dtype=torch.bool, device=device)),
            "edge_attributes": torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            "anchor": (torch.zeros(1, 1, 1, dtype=torch.long, device=device), torch.zeros(1, 1, dtype=torch.bool, device=device))
        }

        return encoder_output, decoder_output, encoder_mask, decoder_mask, batch
