#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from data.parser.to_mrp.abstract_parser import AbstractParser


class EDSParser(AbstractParser):
    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_properties(prediction, output["nodes"])
        output["nodes"] = self.create_anchors(prediction, output["nodes"], at_least_one=self.language != 'zho', single_anchor=self.language != 'zho')
        output["edges"] = self.create_edges(prediction, output["nodes"])
        output["tops"] = self.create_top(prediction, output["nodes"])

        return output

    def label_to_str(self, label, anchors, prediction):
        threshold = min(0.5, anchors.max().item())
        return self.dataset.relative_output_tensor_to_str(
            label - 1,
            (anchors >= threshold).nonzero(as_tuple=False).squeeze(-1),
            prediction["tokens"],
            prediction["lemmas"],
            concat_rules=True,
            ignore_nonalnum=self.language != 'zho',
        )
