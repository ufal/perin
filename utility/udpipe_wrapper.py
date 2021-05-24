#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import requests


class UDPipeWrapper:
    def __init__(self, language="eng"):
        self.url = "http://lindat.mff.cuni.cz/services/udpipe/api/process"
        self.base_request = {
            "tokenizer": "presegmented",
            "tagger": True,
            "model": language,
        }

    def lemmatize(self, sentences):
        request = {"data": '\n'.join(sentences), **self.base_request}
        response = requests.post(self.url, request)

        try:
            response.raise_for_status()
        except requests.HTTPError:
            if len(sentences) == 1:
                raise Exception(f"Unable to parse sentence {sentences[0]} with UDPipe.")
            return self.lemmatize(sentences[:len(sentences) // 2]) + self.lemmatize(sentences[len(sentences) // 2:])

        response = response.json()

        outputs = []
        for sentence in response["result"].split("\n\n"):
            words = [word for word in sentence.split('\n') if len(word) > 0 and not word.startswith('#')]
            words = [word.split('\t') for word in words]
            tokens = [word[1] for word in words]
            lemmas = [word[2] for word in words]

            outputs.append({"tokens": tokens, "lemmas": lemmas})

        return outputs[:-1]
