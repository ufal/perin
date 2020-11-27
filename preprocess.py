#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse

from data.shared_dataset import SharedDataset
from config.params import Params


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--data_directory", type=str, default="/home/samueld/mrp_update/mrp")
    parser.add_argument("--workers", type=int, default=2, help="number of CPU workers per GPU.")
    args = parser.parse_args()

    params = Params()
    params.load(args)
    params.load_state_dict(vars(args))

    return params


if __name__ == "__main__":
    args = parse_arguments()

    dataset = SharedDataset(args)
    dataset.load_datasets(args, 0, 1)
