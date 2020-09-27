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


class UCCAParser(AbstractParser):
    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"], at_least_one=True)

        tops = set()
        for i, node in enumerate(output["nodes"]):
            if node["label"] == "inner":
                tops.add(i)
                del node["anchors"]
            else:
                prediction["edge presence"][i, :] = 0.0

            del node["label"]

        output["edges"] = self.create_edges(prediction, output["nodes"], tops)
        output["tops"] = list(tops)

        return output

    def label_to_str(self, label, anchors, prediction):
        return self.dataset.relative_label_field.vocab.itos[label - 1]

    def create_edges(self, prediction, nodes, tops):
        N = len(nodes)
        node_sets = [{"id": n, "set": set([n])} for n in range(N)]
        _, indices = prediction["edge presence"][:N, :N].reshape(-1).sort(descending=True)
        sources, targets = indices // N, indices % N

        edges = []
        for i in range((N - 1) * N // 2):
            source, target = sources[i].item(), targets[i].item()
            p = prediction["edge presence"][source, target]

            if len(tops) == 1 and p < 0.5 and len(edges) >= N - 1:
                break

            if (node_sets[source]["set"] is node_sets[target]["set"] and p < 0.5) or (len(tops) == 1 and target in tops):
                continue
            elif len(tops) > 1 and target in tops:
                tops.remove(target)

            self.create_edge(source, target, prediction, edges, nodes)

            if node_sets[source]["set"] is not node_sets[target]["set"]:
                from_set = node_sets[source]["set"]
                for n in node_sets[target]["set"]:
                    from_set.add(n)
                    node_sets[n]["set"] = from_set

        return edges

    def create_edge(self, source, target, prediction, edges, nodes):
        labels = self.get_edge_label(prediction, source, target)

        for label in labels:
            edge = {"source": source, "target": target, "label": label}
            attribute = self.get_edge_attribute(prediction, source, target)
            if attribute is not None:
                edge["attributes"] = [attribute]
                edge["values"] = [True]
            edges.append(edge)

    def get_edge_label(self, prediction, source, target):
        options = prediction["edge labels"][source, target, :]
        threshold = min(0.5, options.max().item())  # select at least one
        labels = [self.dataset.edge_label_field.vocab.itos[i] for i, p in enumerate(options) if p >= threshold]
        return labels
