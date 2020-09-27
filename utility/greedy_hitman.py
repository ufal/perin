#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import Counter


def greedy_hitman(data):
    sets = []
    for sentence in data.values():
        for node in sentence["nodes"]:
            rules = {rule["rule"] for rule in node["possible rules"]}
            for s in list(sets):
                if s.issubset(rules):
                    break
                if s.issuperset(rules):
                    sets.remove(s)
            else:
                sets.append(rules)

    rules = []
    while len(sets) > 0:
        rule_counter = Counter()
        for s in sets:
            rule_counter.update(s)
        rule = rule_counter.most_common(1)[0][0]

        sets = [s for s in sets if rule not in s]
        rules.append(rule)

    return rules
