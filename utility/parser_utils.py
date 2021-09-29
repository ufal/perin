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
import math
from collections import Counter
from functools import reduce
import operator
import multiprocessing as mp
import time
from transformers import AutoTokenizer

from utility.label_processor import LabelProcessor
from utility.tokenizer import Tokenizer
from utility.permutation_generator import get_permutations
from utility.bert_tokenizer import bert_tokenizer
from utility.greedy_hitman import greedy_hitman


def load_dataset(path, framework, language=None):
    def condition(s, f, l):
        return ("framework" not in s or f == s["framework"]) and ("framework" in s or f in s["targets"]) and (l is None or s["language"] == l)

    data = {}
    with open(path, encoding="utf8") as f:
        for sentence in f.readlines():
            sentence = json.loads(sentence)
            if condition(sentence, framework, language):
                data[sentence["id"]] = sentence

                if framework == "amr":
                    sentence["input"] = sentence["input"].replace('  ', ' ')
                    sentence["input"] = bytes(sentence["input"], 'utf-8').decode('utf-8', 'ignore')

                if "nodes" not in sentence:
                    continue

                for node in sentence["nodes"]:
                    if "properties" in node:
                        node["properties"] = {prop: node["values"][prop_i] for prop_i, prop in enumerate(node["properties"])}
                        del node["values"]
                    else:
                        node["properties"] = {}

                if "edges" not in sentence:
                    sentence["edges"] = []

    return data


def add_companion(data, path, language: str):
    if path is None:
        add_fake_companion(data, language)
        return

    companion = {}
    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            example = json.loads(line)
            companion[example["id"]] = example

    for sentence in list(data.values()):
        if sentence["id"] not in companion:
            del data[sentence["id"]]
            print(f"WARNING: sentence {sentence['id']} not found in companion, it's omitted from the dataset")

    error_count = 0
    for l in companion.values():
        if l["id"] in data:
            if l["input"].replace(' ', '') != data[l["id"]]["input"].replace(' ', ''):
                print(f"WARNING: sentence {l['id']} not matching companion")
                print(f"original: {data[l['id']]['input']}")
                print(f"companion: {l['input']}")
                print(flush=True)
                del data[l["id"]]
                error_count += 1
                continue

            if language == "zho":
                offset = 0
                for n in l["nodes"]:
                    index = l["input"][offset:].find(n["label"])
                    start = offset + index
                    end = start + len(n["label"])

                    n["anchors"] = [{"from": start, "to": end}]
                    offset = end

            last_start, last_end = None, None
            for i, node in reversed(list(enumerate(l["nodes"]))):
                assert len(node["anchors"]) == 1
                start, end = node["anchors"][0]["from"], node["anchors"][0]["to"]

                if last_start is not None and end - 1 > last_start:
                    node["anchors"][0]["to"] = last_end
                    l["nodes"].pop(i + 1)

                last_start, last_end = start, end

            data[l["id"]]["lemmas"] = [n["values"][n["properties"].index("lemma")] for n in l["nodes"]]
            data[l["id"]]["sentence"] = data[l["id"]]["input"]

            tokens = []
            for n in l["nodes"]:
                assert len(n["anchors"]) == 1
                tokens.append(l["input"][n["anchors"][0]["from"] : n["anchors"][0]["to"]])

            if ''.join(tokens).replace(' ', '') != l["input"].replace(' ', '').replace(' ', '').replace(' ', ''):
                print(f"WARNING: sentence {l['id']} not matching companion after tokenization")
                print(f"companion input: {l['input']}")
                print(f"original: {data[l['id']]['input']}")
                print(f"tokens: {tokens}")
                print(flush=True)
                del data[l["id"]]
                error_count += 1
                continue

            data[l["id"]]["input"] = tokens

    for sentence in list(data.values()):
        try:
            create_token_anchors(sentence)
        except:
            print(f"WARNING: sentence {sentence['id']} not matching companion after anchor computation")
            print(f"tokens: {sentence['input']}")
            print(f"sentence: {sentence['sentence']}")
            print(flush=True)
            del data[sentence["id"]]
            error_count += 1

    print(f"{error_count} erroneously matched sentences with companion")


def add_fake_companion(data, language):
    tokenizer = Tokenizer(data.values(), mode="aggressive")

    for sample in list(data.values()):
        sample["sentence"] = sample["input"]

        token_objects = tokenizer.create_tokens(sample)
        token_objects = [t for t in token_objects if t["token"] is not None]

        tokens = [t["token"]["word"] if isinstance(t["token"], dict) else t["token"] for t in token_objects]
        spans = [t["span"] for t in token_objects]

        sample["input"] = tokens
        sample["lemmas"] = tokens
        sample["token anchors"] = spans


def create_token_anchors(sentence):
    offset = 0
    sentence["token anchors"] = []

    for w in sentence["input"]:
        spaces = 0
        index = sentence["sentence"][offset:].find(w)
        if index != 0 and (index < 0 or not sentence["sentence"][offset:offset + index].isspace()) and offset < len(sentence["sentence"]):
            while offset < len(sentence["sentence"]) and sentence["sentence"][offset] == ' ':
                offset += 1

            index = sentence["sentence"][offset:].replace(' ', '', 1).find(w)
            spaces = 1
            if index < 0:
                raise Exception(f"sentence {sentence['id']} not matching companion after anchor computation.")

        start = offset + index
        end = start + len(w) + spaces

        sentence["token anchors"].append({"from": start, "to": end})
        offset = end


def normalize_properties(data):
    for sentence in data.values():
        properties = []
        node_id = len(sentence["nodes"])
        for node in sentence["nodes"]:
            for relation, value in node["properties"].items():
                nodedized = {
                    "id": node_id,
                    "label": value,
                    "property": True,
                }
                if "anchors" in node:
                    nodedized["anchors"] = node["anchors"]
                properties.append(nodedized)
                sentence["edges"].append({"source": node["id"], "target": node_id, "label": relation, "property": True})

                node_id += 1

            del node["properties"]
        sentence["nodes"] += properties


def node_generator(data):
    for d in data.values():
        for n in d["nodes"]:
            yield n, d


def anchor_ids_from_intervals(data):
    for node, sentence in node_generator(data):
        if "anchors" not in node:
            node["anchors"] = []
        node["anchors"] = sorted(node["anchors"], key=lambda a: (a["from"], a["to"]))
        node["token references"] = set()

        for anchor in node["anchors"]:
            for i, token_anchor in enumerate(sentence["token anchors"]):
                if token_anchor["to"] <= anchor["from"]:
                    continue
                if token_anchor["from"] >= anchor["to"]:
                    break

                node["token references"].add(i)

        node["anchor intervals"] = node["anchors"]
        node["anchors"] = sorted(list(node["token references"]))
        del node["token references"]

    for sentence in data.values():
        sentence["token anchors"] = [[a["from"], a["to"]] for a in sentence["token anchors"]]


def tokenize(data, mode="aggressive"):
    tokenizer = Tokenizer(data.values(), mode=mode)
    for key in data.keys():
        data[key] = tokenizer(data[key])
        data[key] = tokenizer.clean(data[key])


def create_possible_rules(data, applied_function, prune: bool, threads=4):
    print(f"Generating possible rules using {threads} CPUs...", flush=True)
    start_time = time.time()

    # rule_counter = Counter()
    # for node, sentence in node_generator(data):
    #     node["possible rules"] = applied_function(node, sentence)
    #     rule_counter.update((item["rule"] for item in node["possible rules"]))

    # for n, _ in node_generator(data):
    #     n["possible rules"] = [r for r in n["possible rules"] if rule_counter[r["rule"]] > 1 or r["rule"].startswith('a')]

    # return

    # results = [applied_function(node, sentence) for node, sentence in node_generator(data)]
    with mp.Pool(processes=threads) as pool:
        results = pool.starmap(applied_function, node_generator(data))

    rule_domains = {}
    for (node, sentence), rules in zip(node_generator(data), results):
        node["possible rules"] = rules

        if not prune:
            continue

        for rule in rules:
            prefix = rule["rule"][0]
            if prefix != 'l' and prefix != 'd':
                rule_domains[rule["rule"]] = None
                continue

            if prefix == 'd':
                anchors = tuple(sentence["input"][a].lower() for a in rule["anchor"])
            else:
                anchors = tuple(sentence["lemmas"][a].lower() for a in rule["anchor"])

            domain = (prefix, node["label"].lower(), anchors)

            if rule["rule"] not in rule_domains:
                rule_domains[rule["rule"]] = {domain}
            else:
                rule_domains[rule["rule"]].add(domain)

    if not prune:
        return

    print(f"Generated {len(rule_domains)} rules")
    print(f"Pruning unnecessary rules...", flush=True)

    rule_counter = Counter()
    for node, _ in node_generator(data):
        node["possible rules"] = [rule for rule in node["possible rules"] if rule["rule"] in rule_domains]
        for rule in list(node["possible rules"]):
            if rule["rule"][0] != 'l' and rule["rule"][0] != 'd':
                rule_counter.update([rule["rule"]])
                continue
            if rule not in node["possible rules"]:
                continue
            for other_rule in node["possible rules"]:
                if rule["rule"] == other_rule["rule"] or rule_domains[other_rule["rule"]] is None:
                    continue
                domain, other_domain = rule_domains[rule["rule"]], rule_domains[other_rule["rule"]]
                if domain.issubset(other_domain) and (not domain.issubset(other_domain) or len(rule["rule"]) >= len(other_rule["rule"])):
                    node["possible rules"] = [r for r in node["possible rules"] if r["rule"] != rule["rule"]]
                    del rule_domains[rule["rule"]]
                    break
            else:
                rule_counter.update([rule["rule"]])

    print(f"Pruned to {len(rule_counter)} rules")
    print("First 100 most common rules:")
    for m in rule_counter.most_common(100):
        print(f"    {m}")

    print(f"Took {time.time() - start_time} s in total.")


def get_smallest_rule_set(data, approximate: bool):
    print("Solving SAT...", flush=True)

    if approximate:
        return greedy_hitman(data)

    from pysat.examples.hitman import Hitman
    start_time = time.time()

    sets = [{rule["rule"] for rule in node["possible rules"]} for node, _ in node_generator(data)]
    hitman = Hitman(bootstrap_with=sets, solver='g4', htype="sorted")
    best = hitman.get()

    print(f" -> time: {time.time() - start_time}")
    return best


def change_unnecessary_relative_rules(data):
    processor = LabelProcessor()

    label_sets = {}
    for n, _ in node_generator(data):
        for rule in n["possible rules"]:
            rule = rule["rule"]
            if rule not in label_sets:
                label_sets[rule] = set()
            label_sets[rule].add(n["label"].lower())

    for n, _ in node_generator(data):
        for i, rule in enumerate(n["possible rules"]):
            rule = rule["rule"]
            if len(label_sets[rule]) == 1:
                absolute_label = processor.make_absolute_label_rule(n['label'])
                n["possible rules"][i] = {"rule": absolute_label, "anchor": None}


def create_bert_tokens(data, encoder: str):
    tokenizer = AutoTokenizer.from_pretrained(encoder)

    for sentence in data.values():
        to_scatter, bert_input = bert_tokenizer(sentence, tokenizer, encoder)
        sentence["to scatter"] = to_scatter
        sentence["bert input"] = bert_input


def create_edge_permutations(data, similarity_key_f, MAX_LEN=2048):
    def permutation_count(groups):
        return reduce(operator.mul, (math.factorial(len(g)) for g in groups), 1)

    max_n_permutations, max_n_greedy = 1, 0
    for sentence in data.values():
        groups = {}
        for i, node in enumerate(sentence["nodes"]):
            key = similarity_key_f(node)
            if key not in groups:
                groups[key] = [i]
            else:
                groups[key].append(i)

        groups = sorted(groups.values(), key=lambda g: len(g))
        greedy_groups = []
        n_permutations = permutation_count(groups)
        max_n_permutations = max(max_n_permutations, n_permutations)

        while n_permutations > MAX_LEN:
            groups = [[i] for i in groups[-1]] + groups
            greedy_groups.append(groups.pop(-1))
            n_permutations = permutation_count(groups)

        max_n_greedy = max(max_n_greedy, sum(len(g) for g in greedy_groups))
        permutations = get_permutations(groups)
        sentence["edge permutations"] = {"permutations": permutations, "greedy": greedy_groups}

    print(f"Max number of permutations to resolve assignment ambiguity: {max_n_permutations}")
    print(f"... reduced to {min(max_n_permutations, MAX_LEN)} permutations with max of {max_n_greedy} greedily resolved assignments")


def assign_labels_as_best_rules(data, rule_counter):
    for n, _ in node_generator(data):
        possible_rules = set(rule[-1] for rule in n["possible rules"][-1])
        if len(possible_rules) > 0:
            n["label"] = max((rule for rule in possible_rules), key=lambda r: rule_counter[r])
        else:
            n["label"] = "<unk>"


def count_rules(data, label_smoothing):
    rule_counter = Counter({rule[-1]: 0.0 for n, _ in node_generator(data) for rule in n["possible rules"][-1]})

    n_nodes = 0
    for node, _ in node_generator(data):
        rules = {rule[-1] for rule in node["possible rules"][-1]}
        for rule in rules:
            rule_counter[rule] += 1 / len(rules)
        n_nodes += 1

    rule_p = 1 - label_smoothing
    non_rule_p = label_smoothing / (len(rule_counter) - 1)

    for rule in rule_counter.keys():
        rule_counter[rule] = rule_counter[rule] * rule_p + (n_nodes - rule_counter[rule]) * non_rule_p
    return rule_counter


def create_edges(sentence, attributes: bool, label_f=None, normalize=False):
    N = len(sentence["nodes"])

    sentence["edge presence"] = [N, N, []]
    sentence["edge labels"] = [N, N, []]
    sentence["edge attributes"] = [N, N, []]

    for e in sentence["edges"]:
        if normalize and "normal" in e:
            target, source = e["source"], e["target"]
            label = e["normal"].lower()
        else:
            source, target = e["source"], e["target"]
            label = e["label"].lower()

        if label_f is not None:
            label = label_f(label)

        sentence["edge presence"][-1].append((source, target, 1))
        sentence["edge labels"][-1].append((source, target, label))

        if attributes:
            attribute = "<NONE>" if "attributes" not in e else e["attributes"][0]
            sentence["edge attributes"][-1].append((source, target, attribute))

    edge_counter = len(sentence["edge presence"][-1])
    return edge_counter


def create_aligned_rules(data, constrained_anchors: bool):
    for node, sentence in node_generator(data):
        possible_rules = []

        if constrained_anchors:
            anchors = node["anchors"] if len(node["anchors"]) > 0 else range(len(sentence["input"]))
            for anchor in anchors:
                for rule in node["possible rules"]:
                    possible_rules.append((anchor, len(node["possible rules"]), rule["rule"]))
            node["possible rules"] = [len(sentence["input"]), possible_rules]

            continue

        for rule in node["possible rules"]:
            if rule["anchor"] is not None and not rule["rule"].startswith('a'):
                assert len(rule["anchor"]) == 1
                possible_rules.append((rule["anchor"][0], rule["rule"]))
                continue
            for anchor in range(len(sentence["input"])):
                possible_rules.append((anchor, rule["rule"]))
        node["possible rules"] = possible_rules

        possible_rules = []
        for rule in node["possible rules"]:
            possible_rules.append((rule[0], len([r for r in node["possible rules"] if r[0] == rule[0]]), rule[1]))
        node["possible rules"] = [len(sentence["input"]), possible_rules]
