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
import torch.nn.functional as F

from model.head.abstract_head import AbstractHead
from model.module.grad_scaler import scale_grad
from model.module.cross_entropy import cross_entropy
from data.parser.to_mrp.ptg_parser import PTGParser


class PTGHead(AbstractHead):
    def __init__(self, dataset, args, framework, language, initialize: bool):
        config = {
            "label": True,
            "property": True,
            "top": False,
            "edge presence": True,
            "edge label": True,
            "edge attribute": True,
            "anchor": True
        }
        self.property_keys = dataset.property_keys
        super(PTGHead, self).__init__(dataset, args, framework, language, config, initialize)

        self.parser = PTGParser(dataset, language)

    def init_loss_weights(self, config):

        default_weight = 1.0 / (len([v for v in config.values() if v]) + len(self.property_keys) - 1)
        weights = nn.ParameterDict({k: nn.Parameter(torch.tensor([default_weight])) for k, v in config.items() if v})

        del weights["property"]
        for key in self.property_keys:
            weights[f"property {key}"] = nn.Parameter(torch.tensor([default_weight]))

        return weights

    def init_property_classifier(self, dataset, args, config, initialize: bool):
        classifiers = nn.ModuleDict({
            key: nn.Sequential(nn.Dropout(args.dropout_property), nn.Linear(args.hidden_size, len(vocab)))
            for key, vocab in dataset.property_field.vocabs.items()
        })

        if initialize:
            for key, freq in dataset.property_freqs.items():
                classifiers[key][1].bias.data = freq.log()

        return classifiers

    def forward_property(self, decoder_output):
        output = {}
        for key in self.property_keys:
            scaled_decoder_output = scale_grad(decoder_output, self.loss_weights[f"property {key}"])
            output[f"{key}"] = F.log_softmax(self.property_classifier[key](scaled_decoder_output), dim=-1)

        return output

    def inference_property(self, prediction, example_index: int):
        return {key: F.softmax(prediction[key][example_index, :, :], dim=-1).cpu() for key in self.property_keys}

    def loss_property(self, prediction, target, mask):
        loss = {}
        for i, key in enumerate(self.property_keys):
            loss[f"property {key}"] = cross_entropy(prediction["property"][key], target["properties"][:, :, i], mask)
        return loss
