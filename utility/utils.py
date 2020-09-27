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
from PIL import Image


def create_padding_mask(batch_size, total_length, lengths, device):
    mask = torch.arange(total_length, device=device).expand(batch_size, total_length)
    mask = mask >= lengths.unsqueeze(1)  # shape: (B, T)
    return mask


def resize_to_square(image, target_size: int, background_color="white"):
    width, height = image.size
    if width / 2 > height:
        result = Image.new(image.mode, (width, width // 2), background_color)
        result.paste(image, (0, (width // 2 - height) // 2))
        image = result
    elif height * 2 > width:
        result = Image.new(image.mode, (height * 2, height), background_color)
        result.paste(image, ((height * 2 - width) // 2, 0))
        image = result

    image = image.resize([target_size * 2, target_size], resample=Image.BICUBIC)
    return image
