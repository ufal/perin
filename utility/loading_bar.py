#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ["┈", "░", "▒", "▓"]

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length * 4 + 0.5)
        d, r = p // 4, p % 4
        return "┠┈" + d * "█" + ((self.symbols[r]) + max(0, self.length - 1 - d) * "┈" if p < self.length * 4 else "") + "┈┨"
