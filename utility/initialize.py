#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import random
import torch
import datetime
import os


def initialize(args, create_directory: bool, init_wandb: bool, directory_prefix=""):
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if create_directory:
        timestamp = f"{datetime.datetime.today():%m-%d-%y_%H-%M-%S}"
        directory = f"./outputs/{directory_prefix}{timestamp}"
        os.mkdir(directory)
        os.mkdir(f"{directory}/test_predictions")
    else:
        directory = None

    if init_wandb:
        import wandb
        tags = {x for f in args.frameworks for x in f}
        wandb.init(name=args.name, config=args.get_hyperparameters(), project="amr_semantic_parsing", tags=list(tags))
        args.get_hyperparameters().save("config.json")
        wandb.save("config.json")
        print("Connection to Weights & Biases initialized.", flush=True)

    return directory
