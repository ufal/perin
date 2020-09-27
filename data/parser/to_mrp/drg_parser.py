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


class DRGParser(AbstractParser):
    def label_to_str(self, label, anchors, prediction):
        return self.dataset.relative_output_tensor_to_str(
            label - 1,
            anchors,
            prediction["tokens"],
            prediction["lemmas"],
            concat_rules=False,
            num_lemmas=self.language == "deu",
        )

    def create_properties(self, prediction, nodes):
        properties = (prediction["properties"] > 0.5).nonzero(as_tuple=False).squeeze(-1)
        for i in properties:
            nodes[i]["label"] = '"' + nodes[i]["label"] + '"'
        return nodes

    def create_edge(self, source, target, prediction, edges, nodes):
        if nodes[source]["label"] == "<scope>" and nodes[target]["label"] == "<scope>":
            label = self.get_edge_label(prediction, source, target)
            edges.append({"source": source, "target": target, "label": label})
        elif nodes[source]["label"] == "<scope>" and nodes[target]["label"] != "<scope>":
            edges.append({"source": source, "target": target, "label": "in"})
        else:
            label = self.get_edge_label(prediction, source, target)
            new_id = len(nodes)
            edges.append({"source": source, "target": new_id})
            edges.append({"source": new_id, "target": target})
            nodes.append({"id": new_id, "label": label.upper()})

    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_properties(prediction, output["nodes"])
        output["edges"] = self.create_edges(prediction, output["nodes"])
        output["tops"] = self.create_top(prediction, output["nodes"])

        for node in output["nodes"]:
            if node["label"] == "<scope>":
                del node["label"]

        return output
