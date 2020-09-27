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

from model.head.abstract_head import AbstractHead
from model.module.grad_scaler import scale_grad
from model.module.cross_entropy import binary_cross_entropy
from data.parser.to_mrp.ucca_parser import UCCAParser


class UCCAHead(AbstractHead):
    def __init__(self, dataset, args, framework, language, initialize):
        config = {
            "label": True,
            "property": False,
            "top": False,
            "edge presence": True,
            "edge label": True,
            "edge attribute": True,
            "anchor": True
        }
        super(UCCAHead, self).__init__(dataset, args, framework, language, config, initialize)

        self.leaf_id = dataset.label_field.vocab.stoi["leaf"]
        self.parser = UCCAParser(dataset, language)

    def init_label_classifier(self, dataset, args, config, initialize: bool):
        if not config["label"]:
            return None
        classifier = nn.Sequential(nn.Dropout(args.dropout_label), nn.Linear(args.hidden_size, 3))
        if initialize:
            classifier[1].bias.data = dataset.label_freqs.log()
        return classifier

    def forward_label(self, decoder_output, decoder_lens):
        if self.label_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["label"])
        return torch.softmax(self.label_classifier(decoder_output), dim=-1)

    def inference_edge_attribute(self, prediction, example_index: int):
        return (prediction[example_index, :, :, 0] > 0.0).long().cpu()

    def inference_edge_label(self, prediction, example_index: int):
        return prediction[example_index, :, :, :].sigmoid().cpu()

    def loss_edge_presence(self, prediction, target, mask):
        leaf_mask = target["labels"][0] == self.leaf_id  # shape: (B, T_label)
        mask = mask | leaf_mask.unsqueeze(2)
        return {"edge presence": binary_cross_entropy(prediction["edge presence"], target["edge_presence"].float(), mask)}

    def loss_edge_label(self, prediction, target, mask):
        leaf_mask = target["labels"][0] == self.leaf_id  # shape: (B, T_label)
        mask = mask | leaf_mask.unsqueeze(2).unsqueeze(-1)
        return {"edge label": binary_cross_entropy(prediction["edge label"], target["edge_labels"][0].float(), mask)}

    def loss_edge_attribute(self, prediction, target, mask):
        leaf_mask = target["labels"][0] == self.leaf_id  # shape: (B, T_label)
        mask = mask | leaf_mask.unsqueeze(2)
        return {"edge attribute": binary_cross_entropy(prediction["edge attribute"].squeeze(-1), target["edge_attributes"].float(), mask)}
