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
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def match_label(target, matching, shape, device, compute_mask=True, blank_weight=None):
    idx = _get_src_permutation_idx(matching)

    target_classes = torch.zeros(shape, dtype=torch.long, device=device)
    target_classes[idx] = torch.cat([t[J] for t, (_, J) in zip(target, matching)])

    if blank_weight is not None:
        weights = torch.full(target_classes.shape, fill_value=blank_weight, dtype=torch.bool, device=device)
        weights[idx] = 1.0
        return target_classes, weights
    return target_classes


@torch.no_grad()
def match_anchor(anchor, matching, shape, device):
    target, _ = anchor

    idx = _get_src_permutation_idx(matching)
    target_classes = torch.zeros(shape, dtype=torch.long, device=device)
    target_classes[idx] = torch.cat([t[J, :] for t, (_, J) in zip(target, matching)])

    matched_mask = torch.ones(shape[:2], dtype=torch.bool, device=device)
    matched_mask[idx] = False

    return target_classes, matched_mask


@torch.no_grad()
def match_smoothed_label(target, matching, label_smoothing, shape, device, n_queries, blank_weight=None):
    idx = _get_src_permutation_idx(matching)
    target_classes = torch.full(shape, fill_value=label_smoothing / shape[-1], dtype=torch.float, device=device)
    target_classes[:, :, 0] = 1.0 - label_smoothing
    target_classes[idx] = torch.cat([t[J, I // n_queries, :] for t, (I, J) in zip(target, matching)])
    if blank_weight is not None:
        weights = torch.full(target_classes.shape[:2], fill_value=blank_weight, dtype=torch.float, device=device)
        weights[idx] = 1.0
        return target_classes, weights
    return target_classes


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


@torch.no_grad()
def get_matching(cost_matrices):
    output = []
    for cost_matrix in cost_matrices:
        indices = linear_sum_assignment(cost_matrix, maximize=True)
        indices = (torch.tensor(indices[0], dtype=torch.long), torch.tensor(indices[1], dtype=torch.long))
        output.append(indices)

    return output


def sort_by_target(matchings):
    new_matching = []
    for matching in matchings:
        source, target = matching
        target, indices = target.sort()
        source = source[indices]
        new_matching.append((source, target))
    return new_matching


def reorder(hidden, matchings, max_length):
    batch_size, _, hidden_dim = hidden.shape
    matchings = sort_by_target(matchings)

    result = torch.zeros(batch_size, max_length, hidden_dim, device=hidden.device)
    for b in range(batch_size):
        indices = matchings[b][0]
        result[b, : len(indices), :] = hidden[b, indices, :]

    return result
