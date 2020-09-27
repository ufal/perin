#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import utility.parser_utils as utils
from utility.udpipe_wrapper import UDPipeWrapper
from data.parser.from_mrp.abstract_parser import AbstractParser


class RequestParser(AbstractParser):
    def __init__(self, sentences, args, language: str, fields):
        udpipe = UDPipeWrapper(language)

        self.data = {i: {"id": str(i), "sentence": sentence} for i, sentence in enumerate(sentences)}
        for example in self.data.values():
            example["input"], example["lemmas"] = udpipe.lemmatize(example["sentence"])
            utils.create_token_anchors(example)

        utils.tokenize(self.data, mode="aggressive")

        for example in self.data.values():
            example["token anchors"] = [[a["from"], a["to"]] for a in example["token anchors"]]

        utils.create_bert_tokens(self.data, args.encoder)

        super(RequestParser, self).__init__(fields, self.data)
