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


class AMRParser(AbstractParser):
    def parse(self, prediction, approximate_anchors=False):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        if approximate_anchors:
            output["nodes"] = self.create_anchors(prediction, output["nodes"])
        output["nodes"] = self.create_properties(prediction, output["nodes"])
        output["edges"] = self.create_edges(prediction, output["nodes"])
        output["tops"] = self.create_top(prediction, output["nodes"])

        return output

    def label_to_str(self, label, anchors, prediction):
        return self.dataset.relative_output_tensor_to_str(
            label - 1,
            anchors,
            prediction["tokens"],
            prediction["lemmas"],
            concat_rules=False,
        )

    def create_anchors(self, prediction, nodes):
        for i, node in enumerate(nodes):
            node["anchors"] = [prediction["token intervals"][prediction["anchors"][i][0], :]]
            node["anchors"] = [{"from": f.item(), "to": t.item()} for f, t in node["anchors"]]
            node["anchors"] = sorted(node["anchors"], key=lambda a: a["from"])
        return nodes
