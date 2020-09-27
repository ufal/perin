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
from utility.permutation_generator import get_permutations, permute_edges


def test_generator_1():
    groups = [[0], [1], [2]]
    permutations = get_permutations(groups)

    assert len(permutations) == 1
    assert [0, 1, 2] in permutations


def test_generator_2():
    groups = [[0], [1, 3], [2]]
    permutations = get_permutations(groups)

    assert len(permutations) == 2
    assert [0, 1, 2, 3] in permutations
    assert [0, 3, 2, 1] in permutations


def test_generator_3():
    groups = [[0], [1, 2], [3, 4, 5]]
    permutations = get_permutations(groups)

    assert len(permutations) == 12
    assert [0, 1, 2, 3, 4, 5] in permutations
    assert [0, 1, 2, 3, 5, 4] in permutations
    assert [0, 1, 2, 4, 3, 5] in permutations
    assert [0, 1, 2, 4, 5, 3] in permutations
    assert [0, 1, 2, 5, 3, 4] in permutations
    assert [0, 1, 2, 5, 4, 3] in permutations

    assert [0, 2, 1, 3, 4, 5] in permutations
    assert [0, 2, 1, 3, 5, 4] in permutations
    assert [0, 2, 1, 4, 3, 5] in permutations
    assert [0, 2, 1, 4, 5, 3] in permutations
    assert [0, 2, 1, 5, 3, 4] in permutations
    assert [0, 2, 1, 5, 4, 3] in permutations


def test_matching_1():
    permutations = [torch.tensor(get_permutations([[0, 1, 2]]))]
    mask = [torch.zeros(3, dtype=torch.bool)]
    greedy = [[torch.tensor([])]]
    edge_presence_gt = torch.tensor([[1, 0, 0], [0, 0, 1], [1, 1, 0]]).unsqueeze(0)
    batch = {
        "edge_permutations": (permutations, mask, greedy),
        "edge_presence": edge_presence_gt.clone(),
        "edge_attributes": edge_presence_gt.clone(),
        "edge_labels": (edge_presence_gt.clone().unsqueeze(-1), edge_presence_gt.clone()),
        "properties": torch.arange(3).unsqueeze(0).unsqueeze(-1),
        "top": torch.tensor([2]),
    }

    edge_presence_pred = torch.tensor([[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]).unsqueeze(0)
    edge_mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)

    permute_edges(batch, edge_presence_pred, edge_mask)

    should_be = (edge_presence_pred > 0).long()
    assert permutations[0].size(0) == 6
    assert batch["top"][0].item() == 1
    assert (batch["properties"][0, :, 0] == torch.tensor([0, 2, 1])).all(), batch["properties"][0, :, 0].tolist()
    assert (batch["edge_presence"].squeeze(0) == should_be).all(), batch["edge_presence"].squeeze(0).tolist()


def test_matching_2():
    permutations = [torch.tensor(get_permutations([[0, 1], [2], [3], [4]]))]
    mask = [torch.tensor([False, False, True, True, True], dtype=torch.bool)]
    greedy = [[torch.tensor([2, 3, 4])]]
    edge_presence_gt = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]).unsqueeze(0)
    batch = {
        "edge_permutations": (permutations, mask, greedy),
        "edge_presence": edge_presence_gt.clone(),
        "edge_attributes": edge_presence_gt.clone(),
        "edge_labels": (edge_presence_gt.clone().unsqueeze(-1), edge_presence_gt.clone()),
        "properties": torch.arange(5).unsqueeze(0).unsqueeze(-1),
        "top": torch.tensor([3]),
    }

    edge_presence_pred = torch.tensor([[-1.0, -1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0, -1.0]]).unsqueeze(0)
    edge_mask = torch.eye(5, dtype=torch.bool).unsqueeze(0)

    permute_edges(batch, edge_presence_pred, edge_mask)

    should_be = (edge_presence_pred > 0).long()
    assert permutations[0].size(0) == 2
    assert batch["top"][0].item() == 4
    assert (batch["properties"][0, :, 0] == torch.tensor([1, 0, 4, 2, 3])).all(), batch["properties"][0, :, 0].tolist()
    assert (batch["edge_presence"].squeeze(0) == should_be).all(), batch["edge_presence"].squeeze(0).tolist()


def test_matching_3():
    permutations = [torch.tensor(get_permutations([[1], [2], [3], [4], [0, 5]]))]
    mask = [torch.tensor([False, True, True, False, True, False], dtype=torch.bool)]
    greedy = [[torch.tensor([1, 2, 4])]]
    edge_presence_gt = torch.tensor([[0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]).unsqueeze(0)
    batch = {
        "edge_permutations": (permutations, mask, greedy),
        "edge_presence": edge_presence_gt.clone(),
        "edge_attributes": edge_presence_gt.clone(),
        "edge_labels": (edge_presence_gt.clone().unsqueeze(-1), edge_presence_gt.clone()),
        "properties": torch.arange(6).unsqueeze(0).unsqueeze(-1),
        "top": torch.tensor([1]),
    }

    edge_presence_pred = torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0], [1.0, 1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0]]).unsqueeze(0)
    edge_mask = torch.eye(6, dtype=torch.bool).unsqueeze(0)

    permute_edges(batch, edge_presence_pred, edge_mask)

    should_be = (edge_presence_pred > 0).long()
    assert permutations[0].size(0) == 2
    assert batch["top"][0].item() == 4
    assert (batch["properties"][0, :, 0] == torch.tensor([5, 4, 2, 3, 1, 0])).all(), batch["properties"][0, :, 0].tolist()
    assert (batch["edge_presence"].squeeze(0) == should_be).all(), batch["edge_presence"].squeeze(0).tolist()


def test_matching_4():
    permutations = [torch.tensor(get_permutations([[0], [1], [2], [3], [4], [5]]))]
    mask = [torch.tensor([True, True, True, False, True, True], dtype=torch.bool)]
    greedy = [[torch.tensor([0, 5]), torch.tensor([1, 4])]]
    edge_presence_gt = torch.tensor([[0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]).unsqueeze(0)
    batch = {
        "edge_permutations": (permutations, mask, greedy),
        "edge_presence": edge_presence_gt.clone(),
        "edge_attributes": edge_presence_gt.clone(),
        "edge_labels": (edge_presence_gt.clone().unsqueeze(-1), edge_presence_gt.clone()),
        "properties": torch.arange(6).unsqueeze(0).unsqueeze(-1),
        "top": torch.tensor([1]),
    }

    edge_presence_pred = torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0], [1.0, 1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0]]).unsqueeze(0)
    edge_mask = torch.eye(6, dtype=torch.bool).unsqueeze(0)

    permute_edges(batch, edge_presence_pred, edge_mask)

    should_be = (edge_presence_pred > 0).long()
    assert permutations[0].size(0) == 1
    assert batch["top"][0].item() == 4
    assert (batch["properties"][0, :, 0] == torch.tensor([5, 4, 2, 3, 1, 0])).all(), batch["properties"][0, :, 0].tolist()
    assert (batch["edge_presence"].squeeze(0) == should_be).all(), batch["edge_presence"].squeeze(0).tolist()
