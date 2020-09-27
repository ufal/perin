#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from model.head.abstract_head import AbstractHead
from data.parser.to_mrp.drg_parser import DRGParser
from model.module.cross_entropy import binary_cross_entropy


class DRGHead(AbstractHead):
    def __init__(self, dataset, args, framework, language, initialize: bool):
        config = {
            "label": True,
            "property": True,
            "top": True,
            "edge presence": True,
            "edge label": True,
            "edge attribute": False,
            "anchor": False
        }
        super(DRGHead, self).__init__(dataset, args, framework, language, config, initialize)
        self.parser = DRGParser(dataset, language)

    def loss_edge_label(self, prediction, target, mask):
        mask = mask | target["edge_labels"][1].unsqueeze(-1)
        return {"edge label": binary_cross_entropy(prediction["edge label"], target["edge_labels"][0].float(), mask)}
