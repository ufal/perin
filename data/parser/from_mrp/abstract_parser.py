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
from data.parser.json_parser import example_from_json


class AbstractParser(torch.utils.data.Dataset):
    def __init__(self, fields, data, filter_pred=None):
        super(AbstractParser, self).__init__()

        self.examples = [example_from_json(o, fields) for o in data.values()]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        if filter_pred is not None:
            make_list = isinstance(self.examples, list)
            self.examples = filter(filter_pred, self.examples)
            if make_list:
                self.examples = list(self.examples)

        self.fields = dict(fields)

        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    def get_examples(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
