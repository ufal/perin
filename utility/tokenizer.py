#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import re
from utility.xml_parser import XMLParser
import pyonmttok


class Tokenizer:
    def __init__(self, data, mode="aggresive"):
        self.parser = XMLParser()
        self.html_regex = re.compile(r"<\s*([a-zA-Z]+)[^>]*>(.*?)<\s*\/\s*\1>")
        self.wrong_count = 0
        self.tokenizer = pyonmttok.Tokenizer(mode)

    def create_tokens(self, sentence):
        new_tokens = [{"token": None, "span": {"from": float("-inf"), "to": 0}}]

        offset = 0
        spans = []
        for m in self.html_regex.finditer(sentence["sentence"]):
            span = m.span()
            space = span[0] - offset
            if space > 1 or space == 1 and sentence["sentence"][offset] != " ":
                spans.append({"span": {"from": offset, "to": offset + space}, "is_html": False})

            spans.append({"span": {"from": span[0], "to": span[1]}, "is_html": True})
            offset = span[1]

        space = len(sentence["sentence"]) - offset
        if space > 1 or space == 1 and sentence["sentence"][offset] != " ":
            spans.append({"span": {"from": offset, "to": offset + space}, "is_html": False})

        offset = 0
        for m in spans:
            span = m["span"]

            space = span["from"] - offset
            if space > 1 or space == 1 and sentence["sentence"][offset] != " ":
                new_tokens.append({"token": None, "span": {"from": offset, "to": offset + space}})

            offset = span["from"]
            if m["is_html"]:
                for o in self.parser.feed(sentence["sentence"][span["from"]: span["to"]]):
                    space = sentence["sentence"][offset:].find(o["word"])

                    if space == -1:
                        o["word"] = o["word"].replace('"', "&quot;")  # hack to make it work somehow
                        space = sentence["sentence"][offset:].find(o["word"])
                        if space == -1:
                            self.wrong_count += 1
                            space = 0

                    if space > 1 or space == 1 and sentence["sentence"][offset] != " ":
                        new_tokens.append({"token": None, "span": {"from": offset, "to": offset + space}})

                    start = offset + space
                    end = start + len(o["word"])
                    new_tokens.append({"token": o, "span": {"from": start, "to": end}})
                    offset = end

            else:
                tokens = self.tokenizer.tokenize(sentence["sentence"][span["from"]: span["to"]])[0]
                for token in tokens:
                    if token != "":
                        start = offset + sentence["sentence"][offset:].find(token)
                        end = start + len(token)
                        new_tokens.append(
                            {"token": {"word": token, "lemma": None}, "span": {"from": start, "to": end}}
                        )
                        offset = end

        space = len(sentence["sentence"]) - offset
        if space > 1 or space == 1 and sentence["sentence"][offset] != " ":
            new_tokens.append({"token": None, "span": {"from": offset, "to": offset + space}})

        new_tokens.append({"token": None, "span": {"from": len(sentence["sentence"]), "to": float("inf")}})
        return new_tokens

    def __call__(self, sentence):
        new_tokens = self.create_tokens(sentence)

        offset, increase = 0, 0
        new_input, new_lemmas, new_spans = [], [], []

        for i, input in enumerate(sentence["input"]):
            derived_tokens = []
            orig_from, orig_to = (
                sentence["token anchors"][i]["from"],
                sentence["token anchors"][i]["to"],
            )

            while new_tokens[offset]["span"]["to"] <= orig_from:
                offset += 1

            first = True

            while True:
                new_token = new_tokens[offset]
                new_from, new_to = new_token["span"]["from"], new_token["span"]["to"]

                if new_from >= orig_to:
                    if first:
                        derived_tokens.append(
                            {
                                "word": input,
                                "lemma": sentence["lemmas"][i],
                                "span": sentence["token anchors"][i],
                            }
                        )
                    break

                if new_from <= orig_from:
                    start = orig_from
                    end = orig_to if new_to > orig_to else new_to
                elif new_from > orig_from:
                    start = new_from
                    end = orig_to if new_to > orig_to else new_to

                word = sentence["sentence"][start:end]

                if new_token["token"] is not None and new_token["token"]["lemma"] is not None:
                    lemma = new_token["token"]["lemma"]
                elif sentence["lemmas"][i].lower() == input.lower():
                    lemma = word.lower()
                else:
                    lemma = sentence["lemmas"][i]

                derived_tokens.append({"word": word, "lemma": lemma, "span": {"from": start, "to": end}})

                first = False
                if new_to <= orig_to:
                    offset += 1
                else:
                    break

            for j, t in enumerate(derived_tokens):
                new_input.append(t["word"])
                new_lemmas.append(t["lemma"])
                new_spans.append(t["span"])

                if "nodes" not in sentence:
                    continue

                for n in sentence["nodes"]:
                    if "anchor" not in n:
                        continue
                    for k, a in enumerate(n["anchor"]):
                        if j > 0 and a >= i + increase + j:
                            n["anchor"][k] += 1
                    if i + increase in n["anchor"]:
                        n["anchor"].append(i + increase + j)

            increase += len(derived_tokens) - 1

        sentence["input"] = new_input
        sentence["lemmas"] = new_lemmas
        sentence["token anchors"] = new_spans

        if "nodes" not in sentence:
            return sentence

        for n in sentence["nodes"]:
            if "anchor" not in n:
                continue
            n["anchor"] = list(sorted(set(n["anchor"])))
            n["anchor tokens"] = [sentence["input"][a] for a in n["anchor"]]

        return sentence

    def clean(self, sentence):
        to_delete = []
        for i, word in enumerate(sentence["input"]):
            word = re.sub(r"\s+", "", word, flags=re.UNICODE)
            sentence["input"][i] = word
            if len(word) == 0:
                to_delete.append(i)

        for i in to_delete[::-1]:
            del sentence["input"][i]
            del sentence["lemmas"][i]
            del sentence["token anchors"][i]

            if "nodes" not in sentence:
                continue

            for n in sentence["nodes"]:
                if "anchor" not in n:
                    continue
                if i in n["anchor"]:
                    n["anchor"].remove(i)
                for k, a in enumerate(n["anchor"]):
                    if a > i:
                        n["anchor"][k] -= 1

        return sentence
