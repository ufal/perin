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

from collections import Counter
from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils
from utility.label_processor import LabelProcessor


class AMRParser(AbstractParser):
    def __init__(self, args, framework: str, language: str, part: str, fields, precomputed_dataset=None, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data[(framework, language)] if part == "training" else args.validation_data[(framework, language)]

        self.framework = framework
        self.language = language

        cache_path = f"{path}_cache"
        if not os.path.exists(cache_path):
            self.initialize(args, path, cache_path, args.companion_data[(framework, language)], precomputed_dataset=precomputed_dataset)

        print("Loading the cached dataset")

        self.data = {}
        with io.open(cache_path, encoding="utf8") as reader:
            for line in reader:
                sentence = json.loads(line)
                self.data[sentence["id"]] = sentence

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0

        for node, sentence in utils.node_generator(self.data):
            self.node_counter += 1
            node["properties"] = {"transformed": int("property" in node)}

        utils.create_aligned_rules(self.data, constrained_anchors=False)
        self.rule_counter = utils.count_rules(self.data, args.label_smoothing)

        utils.create_bert_tokens(self.data, args.encoder)
        utils.assign_labels_as_best_rules(self.data, self.rule_counter)
        utils.create_edge_permutations(self.data, AMRParser.node_similarity_key)

        # create edge vectors

        for sentence in self.data.values():
            N = len(sentence["nodes"])

            edge_count = utils.create_edges(sentence, attributes=False, normalize=args.normalize)
            self.edge_counter += edge_count
            self.no_edge_counter += N * (N - 1) - edge_count

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            sentence["id"] = [sentence["id"]]
            sentence["top"] = sentence["tops"][0]
            sentence["token anchors"] = [[]]

        self.anchor_freq = 0.0
        self.input_count = sum(len(sentence["input"]) for sentence in self.data.values())

        super(AMRParser, self).__init__(fields, self.data, filter_pred)

    def initialize(self, args, raw_path, cache_path, companion_path, precomputed_dataset=None):
        print("Caching the dataset...\n", flush=True)

        data = utils.load_dataset(raw_path, framework=self.framework)

        utils.add_companion(data, companion_path, self.language)
        utils.normalize_properties(data)
        utils.tokenize(data, mode="aggressive")

        for node, sentence in utils.node_generator(data):
            node["anchors"] = list(range(len(sentence["input"])))
            assert '│' not in node["label"]

        # create relative labels

        utils.create_possible_rules(data, AMRParser._create_possible_rules, prune=False)
        if precomputed_dataset is None:
            rule_set = utils.get_smallest_rule_set(data, approximate=False)
        else:
            rule_set = set(r[2] for e in precomputed_dataset.values() for n in e["nodes"] for r in n["possible rules"][1])

        print(f" -> # relative labels: {len(rule_set)}\n", flush=True)

        for n, _ in utils.node_generator(data):
            n["possible rules"] = [item for item in n["possible rules"] if item["rule"] in rule_set]

        if precomputed_dataset is None:
            utils.change_unnecessary_relative_rules(data)

        rule_counter = Counter()
        for n, d in utils.node_generator(data):
            rule_counter.update((item["rule"] for item in n["possible rules"]))

        for rule, count in rule_counter.most_common():
            print(f"- `{rule}`: {count}")
        print(flush=True)

        with open(cache_path, "w", encoding="utf8") as f:
            for example in data.values():
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _create_possible_rules(node, sentence):
        processor = LabelProcessor()
        rules = []

        for anchor in node["anchors"]:
            _rules = processor.gen_all_label_rules(
                sentence["input"][anchor:],
                sentence["lemmas"][anchor:],
                node["label"],
                separators=['', '-'],
                rule_classes=["relative_forms", "relative_lemmas", "numerical_divide", "concatenate"],
                concat=False,
                allow_copy=False,
            )
            for rule in set(_rules):
                rules.append({"rule": rule, "anchor": [anchor]})

        rules.append({"rule": processor.make_absolute_label_rule(node["label"]), "anchor": None})

        return rules

    @staticmethod
    def node_similarity_key(node):
        if node["label"].startswith('a'):
            return node["label"]
        return node["id"]
