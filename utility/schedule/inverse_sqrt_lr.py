#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class InverseSqrtLr:
    def __init__(self, param_group, learning_rate: float, warmup_steps: int, delay_steps: int):
        self.warmup_steps = warmup_steps
        self.base = learning_rate
        self.decay_factor = learning_rate * warmup_steps ** 0.5
        self.steps = -delay_steps
        self.param_group = param_group

        self.__call__(0)

    def __call__(self, _):
        self.steps += 1

        if self.steps < 0:
            lr = 0.0
        elif self.steps < self.warmup_steps:
            lr = self.base / self.warmup_steps * self.steps
        else:
            lr = self.decay_factor * self.steps ** -0.5

        self.param_group["lr"] = lr

    def lr(self) -> float:
        return self.param_group["lr"]
