#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json
import io
import os
import os.path

from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class UCCAParser(AbstractParser):
    def __init__(self, args, framework: str, language: str, part: str, fields, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data[(framework, language)] if part == "training" else args.validation_data[(framework, language)]

        self.framework = framework
        self.language = language

        cache_path = f"{path}_cache"
        if not os.path.exists(cache_path):
            self.initialize(args, path, cache_path, args.companion_data[(framework, language)])

        print("Loading the cached dataset")

        self.data = {}
        with io.open(cache_path, encoding="utf8") as reader:
            for line in reader:
                sentence = json.loads(line)
                self.data[sentence["id"]] = sentence

        utils.create_bert_tokens(self.data, args.encoder)
        utils.create_edge_permutations(self.data, UCCAParser.node_similarity_key)

        for sentence in self.data.values():
            sentence["top"] = sentence["tops"][0]

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, n_node_token_pairs = 0, 0

        for node, sentence in utils.node_generator(self.data):
            self.node_counter += 1
            node["properties"] = {"dummy": 0}

            node["possible rules"] = [len(sentence["input"]), []]
            assert len(node["anchors"]) > 0
            for anchor in node["anchors"]:
                node["possible rules"][-1].append((anchor, 1, node["label"]))

        self.rule_counter = utils.count_rules(self.data, args.label_smoothing)

        # create edge vectors

        for sentence in self.data.values():
            N = len(sentence["nodes"])

            edge_count = utils.create_edges(sentence, attributes=True)
            self.edge_counter += edge_count
            self.no_edge_counter += N * (N - 1) - edge_count

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            for i, node in enumerate(sentence["nodes"]):
                for anchor in node["anchors"]:
                    sentence["anchor edges"][-1].append((i, anchor))

                anchor_count += len(node["anchors"])
                n_node_token_pairs += len(sentence["input"])

            sentence["id"] = [sentence["id"]]
            sentence["top"] = 0
            sentence["properties"] = []

        self.input_count = sum(len(sentence["input"]) for sentence in self.data.values())
        self.anchor_freq = anchor_count / n_node_token_pairs

        super(UCCAParser, self).__init__(fields, self.data, filter_pred)

    def initialize(self, args, raw_path, cache_path, companion_path):
        print("Caching the dataset...", flush=True)

        data = utils.load_dataset(raw_path, framework=self.framework)
        utils.add_companion(data, companion_path, self.language)
        utils.tokenize(data, mode="aggressive")

        # divide into leaf in inner nodes and induce inner anchors

        for sentence in data.values():
            out_edges = [[] for _ in sentence["nodes"]]
            in_edges = [[] for _ in sentence["nodes"]]

            for edge in sentence["edges"]:
                out_edges[edge["source"]].append(edge["target"])
                in_edges[edge["target"]].append(edge["source"])

            leaves = []
            for node in sentence["nodes"]:
                if len(out_edges[node["id"]]) == 0:
                    node["label"] = "leaf"
                    leaves.append(node["id"])
                else:
                    node["label"] = "inner"
                node["label"] = "leaf" if len(out_edges[node["id"]]) == 0 else "inner"

                if "anchors" not in node:
                    node["anchors"] = []

                node["anchors"] = {(a["from"], a["to"]): a for a in node["anchors"]}
                node["parents"] = set()

            depth = 0
            layer = [sentence["tops"][0]]
            while len(layer) > 0:
                new_layer = []
                for n in layer:
                    sentence["nodes"][n]["depth"] = depth
                    for child in out_edges[n]:
                        if n not in sentence["nodes"][child]["parents"]:
                            sentence["nodes"][child]["parents"].add(n)
                            new_layer.append(child)
                depth += 1
                layer = new_layer

            layer = leaves
            while len(layer) > 0:
                new_layer = []
                for n in layer:
                    for parent in in_edges[n]:
                        for a in sentence["nodes"][n]["anchors"].keys():
                            if a not in sentence["nodes"][parent]["anchors"]:
                                new_layer.append(parent)
                                break
                        sentence["nodes"][parent]["anchors"].update(sentence["nodes"][n]["anchors"])
                layer = new_layer

            for node in sentence["nodes"]:
                node["anchors"] = list(node["anchors"].values())
                del node["parents"]

        utils.anchor_ids_from_intervals(data)

        with open(cache_path, "w", encoding="utf8") as f:
            for sentence in data.values():
                json.dump(sentence, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
