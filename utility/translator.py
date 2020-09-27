#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
from transformers import MarianTokenizer, MarianMTModel


class Translator:
    def __init__(self, source_language: str, target_language: str):
        name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarianMTModel.from_pretrained(name).to(self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(name)

    def translate(self, words):
        batch = self.tokenizer.prepare_translation_batch(src_texts=words).to(self.device)
        gen = self.model.generate(**batch).to(torch.device("cpu"))
        translated_words = self.tokenizer.batch_decode(gen, skip_special_tokens=True)

        return translated_words
