#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import bisect
import torch


class ConcatDataset(torch.utils.data.Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'

        self.datasets = list(datasets)
        self.fields = [{name: field for (name, field) in dataset.fields.items() if field is not None} for dataset in datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        item = self.datasets[dataset_idx][sample_idx]
        processed_item = {}
        for (name, field) in self.fields[dataset_idx].items():
            if field is not None:
                processed_item[name] = field.process(getattr(item, name), device=None)

        processed_item["framework"] = torch.tensor(dataset_idx, dtype=torch.long)
        return processed_item

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes
