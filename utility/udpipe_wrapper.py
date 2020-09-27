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
        url = "http://lindat.mff.cuni.cz/services/udpipe/api/process"
        self.request = f"{url}?tokenizer&tagger&model={language}&data={{0}}"

    def lemmatize(self, sentence):
        response = requests.get(self.request.format(sentence)).json()

        lines = response["result"].split('\n')
        words = [line.split('\t') for line in lines if len(line) > 0 and not line.startswith('#')]
        tokens = [word[1] for word in words]
        lemmas = [word[2] for word in words]

        return tokens, lemmas
