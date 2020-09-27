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
from utility.frame_predictor import FramePredictor


class PTGParser(AbstractParser):
    def __init__(self, dataset, language):
        super().__init__(dataset, language)
        self.frame_predictor = FramePredictor(language)

    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"])
        output["nodes"] = self.create_properties(prediction, output["nodes"])
        output["edges"] = self.create_edges(prediction, output["nodes"])
        output["tops"] = self.create_top(prediction, output["nodes"])

        return output

    def label_to_str(self, label, anchors, prediction):
        threshold = min(0.5, anchors.max().item())
        label = self.dataset.relative_output_tensor_to_str(
            label - 1,
            (anchors >= threshold).nonzero(as_tuple=False).squeeze(-1),
            prediction["tokens"],
            prediction["lemmas"],
            concat_rules=True,
        )
        label = label.replace('/', '\\/')
        return label

    def create_properties(self, prediction, nodes):
        for node in nodes:
            if node["label"].lower() == "<top>":
                del node["label"], node["anchors"]
                continue

            node_properties = {}
            for key in prediction["properties"].keys():
                prop = self.dataset.property_field.vocabs[key].itos[prediction["properties"][key][node["id"], :].argmax()]
                if key == "frame" and prop != "<NONE>":
                    frame_probs = {
                        self.dataset.property_field.vocabs[key].itos[i]: p
                        for i, p in enumerate(prediction["properties"][key][node["id"], :])
                    }
                    prop = self.frame_predictor.predict(node["label"].lower(), frame_probs)

                if prop != "<NONE>":
                    node_properties[key] = prop

            if len(node_properties) > 0:
                node["properties"] = list(node_properties.keys())
                node["values"] = list(node_properties.values())

        return nodes

    def create_top(self, prediction, nodes):
        for i, node in enumerate(nodes):
            if "label" not in node:
                return [i]
        return [0]
