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
import itertools
from model.module.cross_entropy import binary_cross_entropy


@torch.no_grad()
def permute_edges(batch, edge_presence, edge_mask):
    all_permutations, all_masks, all_greedies = batch["edge_permutations"]

    for b in range(edge_presence.size(0)):
        permutations = all_permutations[b]
        mask = all_masks[b].unsqueeze(0) | all_masks[b].unsqueeze(1)
        greedies = all_greedies[b]
        N = permutations.size(1)

        if permutations.size(0) > 1:
            edge_presence_gt = many_permute_edges(batch["edge_presence"][b, :N, :N].float(), permutations)
            edge_presence_pred = edge_presence[b, :N, :N].expand_as(edge_presence_gt)
            edge_presence_mask = edge_mask[b, :N, :N].expand_as(edge_presence_gt) | mask
            scores = binary_cross_entropy(edge_presence_pred, edge_presence_gt, edge_presence_mask, reduction=False)
            best_index = scores.sum(dim=[1, 2]).argmin()
            permutation = permutations[best_index, :]

            batch["edge_presence"][b, :N, :N] = single_permute_edges(batch["edge_presence"][b, :N, :N], permutation)
            batch["edge_attributes"][b, :N, :N] = single_permute_edges(batch["edge_attributes"][b, :N, :N], permutation)
            batch["edge_labels"][0][b, :N, :N, :] = single_permute_edges(batch["edge_labels"][0][b, :N, :N, :], permutation)
            batch["edge_labels"][1][b, :N, :N] = single_permute_edges(batch["edge_labels"][1][b, :N, :N], permutation)
            batch["properties"][b, :N, :] = batch["properties"][b, permutation, :]
            batch["top"][b] = permutation[batch["top"][b]]

        for greedy in greedies:
            greedy = greedy.cpu().tolist()
            for i in list(greedy):
                mask[i, :], mask[:, i] = False, False

                edge_presence_original = batch["edge_presence"][b, :N, :N].float()
                edge_presence_gt = edge_presence_original.expand(len(greedy), -1, -1).contiguous()
                edge_presence_pred = edge_presence[b, :N, :N].expand_as(edge_presence_gt)
                edge_presence_mask = edge_mask[b, :N, :N].expand_as(edge_presence_gt) | mask.unsqueeze(0)

                for j, g in enumerate(greedy):
                    edge_presence_gt[j, i, :] = edge_presence_original[g, :]
                    edge_presence_gt[j, :, i] = edge_presence_original[:, g]

                scores = binary_cross_entropy(edge_presence_pred, edge_presence_gt, edge_presence_mask, reduction=False)
                scores = scores.sum(dim=[1, 2])
                best_node = greedy[scores.argmin()]
                greedy = greedy[1:]

                permutation = torch.arange(N, dtype=torch.long, device=edge_presence_original.device)
                permutation[i], permutation[best_node] = best_node, i

                batch["edge_presence"][b, :N, :N] = single_permute_edges(batch["edge_presence"][b, :N, :N], permutation)
                batch["edge_attributes"][b, :N, :N] = single_permute_edges(batch["edge_attributes"][b, :N, :N], permutation)
                batch["edge_labels"][0][b, :N, :N, :] = single_permute_edges(batch["edge_labels"][0][b, :N, :N, :], permutation)
                batch["edge_labels"][1][b, :N, :N] = single_permute_edges(batch["edge_labels"][1][b, :N, :N], permutation)
                batch["properties"][b, :N, :] = batch["properties"][b, permutation, :]
                batch["top"][b] = permutation[batch["top"][b]]


def many_permute_edges(edges, permutations):
    assert edges.dim() == 2

    index_1 = permutations.unsqueeze(2).expand(-1, -1, edges.size(1))
    index_2 = permutations.unsqueeze(1).expand(-1, edges.size(0), -1)

    edges = edges.expand(permutations.size(0), -1, -1).gather(1, index_1).gather(2, index_2)
    return edges


def single_permute_edges(edges, permutation):
    index_0 = permutation.unsqueeze(1).expand(-1, edges.size(1))
    index_1 = permutation.unsqueeze(0).expand(edges.size(0), -1)

    if edges.dim() == 3:
        index_0 = index_0.unsqueeze_(-1).expand(-1, -1, edges.size(-1))
        index_1 = index_1.unsqueeze_(-1).expand(-1, -1, edges.size(-1))
    else:
        assert edges.dim() == 2

    edges = edges.gather(0, index_0).gather(1, index_1)
    return edges


def get_permutations(groups):
    n = sum(len(g) for g in groups)
    permutations = [list(range(n))]

    for group in groups:
        if len(group) == 1:
            continue

        pi = itertools.permutations(group)
        new_permutations = []

        for p_2 in pi:
            for p_1 in permutations:
                new_permutations.append(p_1.copy())
                for i, p in enumerate(p_2):
                    new_permutations[-1][group[i]] = p

        permutations = new_permutations

    return permutations
