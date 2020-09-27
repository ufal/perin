#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from html.parser import HTMLParser
import pyonmttok
import re


class XMLParser(HTMLParser):
    def __init__(self):
        self.tokenizer = pyonmttok.Tokenizer("aggressive")
        self.url_regex = re.compile(r"(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?")

        super(XMLParser, self).__init__()

    def feed(self, data):
        self.buffer = []
        super().feed(data)
        return self.buffer

    def handle_starttag(self, tag, attrs):
        self.buffer.append({"word": f"<", "lemma": "<start-tag>"})
        self.buffer.append({"word": tag, "lemma": tag})
        for attr in attrs:
            self.buffer.append(
                {"word": attr[0], "lemma": "<attribute>"}
            )

            if attr[0] == "href":
                self.buffer.append(
                    {"word": attr[1], "lemma": "<url-entity>"}
                )
            else:
                self.buffer.append({"word": attr[1], "lemma": attr[1]})

    def handle_endtag(self, tag):
        self.buffer.append({"word": f"</", "lemma": "<end-tag>"})
        self.buffer.append({"word": tag, "lemma": tag})
        self.buffer.append({"word": ">", "lemma": "<end-tag>"})

    def handle_data(self, data):
        if self.url_regex.fullmatch(data):
            self.buffer.append({"word": data, "lemma": "<url-entity>"})
            return

        for token in self.tokenizer.tokenize(data)[0]:
            self.buffer.append({"word": token, "lemma": None})
