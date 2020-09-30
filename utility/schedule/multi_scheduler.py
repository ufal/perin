#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from utility.schedule.inverse_sqrt_lr import InverseSqrtLr


def multi_scheduler_wrapper(optimizer, args):
    return MultiScheduler(
        [
            InverseSqrtLr(
                optimizer.param_groups[i], args.encoder_learning_rate * (args.layerwise_lr_decay ** i), args.warmup_steps, args.encoder_delay_steps
            )
            for i in range(len(optimizer.param_groups) - 1)
        ]
        +
        [
            InverseSqrtLr(optimizer.param_groups[-1], args.decoder_learning_rate, args.warmup_steps, args.decoder_delay_steps)
        ]
    )


class MultiScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def __call__(self, epoch):
        for scheduler in self.schedulers:
            scheduler(epoch)

    def lr(self) -> float:
        return [scheduler.lr() for scheduler in self.schedulers]
