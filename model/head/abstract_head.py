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
import torch.nn as nn
import torch.nn.functional as F

from model.module.edge_classifier import EdgeClassifier
from model.module.anchor_classifier import AnchorClassifier
from model.module.padding_packer import PaddingPacker
from model.module.mixture_of_softmaxes import MixtureOfSoftmaxes
from model.module.grad_scaler import scale_grad
from model.module.cross_entropy import multi_label_cross_entropy, cross_entropy, binary_cross_entropy
from utility.hungarian_matching import get_matching, reorder, match_smoothed_label, match_anchor
from utility.utils import create_padding_mask
from utility.permutation_generator import permute_edges


class AbstractHead(nn.Module):
    def __init__(self, dataset, args, framework, language, config, initialize: bool):
        super(AbstractHead, self).__init__()

        self.loss_weights = self.init_loss_weights(config)

        self.edge_classifier = self.init_edge_classifier(dataset, args, config, initialize)
        self.label_classifier = self.init_label_classifier(dataset, args, config, initialize)
        self.top_classifier = self.init_top_classifier(dataset, args, config, initialize)
        self.property_classifier = self.init_property_classifier(dataset, args, config, initialize)
        self.anchor_classifier = self.init_anchor_classifier(dataset, args, config, initialize)

        self.query_length = args.query_length
        self.label_smoothing = args.label_smoothing
        self.focal = args.focal
        self.blank_weight = args.blank_weight
        self.dataset = dataset
        self.framework = framework
        self.language = language

    def forward(self, encoder_output, decoder_output, encoder_mask, decoder_mask, batch):
        output = {}

        decoder_lens = self.query_length * batch["every_input"][1]
        output["label"] = self.forward_label(decoder_output, decoder_lens)
        output["anchor"] = self.forward_anchor(decoder_output, encoder_output, encoder_mask)  # shape: (B, T_l, T_w)

        cost_matrices = self.create_cost_matrices(output, batch, decoder_lens)
        matching = get_matching(cost_matrices)
        decoder_output = reorder(decoder_output, matching, batch["properties"].size(1))

        output["property"] = self.forward_property(decoder_output)
        output["top"] = self.forward_top(decoder_output)
        output["edge presence"], output["edge label"], output["edge attribute"] = self.forward_edge(decoder_output)

        return self.loss(output, batch, matching, decoder_mask)

    def predict(self, encoder_output, decoder_output, encoder_mask, decoder_mask, batch, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size = every_input.size(0)

        label_pred = self.forward_label(decoder_output, decoder_lens)
        anchor_pred = self.forward_anchor(decoder_output, encoder_output, encoder_mask)  # shape: (B, T_l, T_w)

        labels, anchors = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        for b in range(batch_size):
            label_indices = self.inference_label(label_pred[b, :decoder_lens[b], :]).cpu()
            for t in range(label_indices.size(0)):
                relative_label_index = label_indices[t].item()
                if relative_label_index == 0:
                    continue

                decoder_output[b, len(labels[b]), :] = decoder_output[b, t, :]

                labels[b].append(relative_label_index)
                if anchor_pred is None:
                    anchors[b].append(list(range(t // self.query_length, word_lens[b])))
                else:
                    anchors[b].append(self.inference_anchor(anchor_pred[b, t, :word_lens[b]]).cpu())

        decoder_output = decoder_output[:, : max(len(l) for l in labels), :]

        properties = self.forward_property(decoder_output)
        tops = self.forward_top(decoder_output)
        edge_presence, edge_labels, edge_attributes = self.forward_edge(decoder_output)

        outputs = [
            self.parser.parse(
                {
                    "labels": labels[b],
                    "anchors": anchors[b],
                    "properties": self.inference_property(properties, b),
                    "tops": self.inference_top(tops, b),
                    "edge presence": self.inference_edge_presence(edge_presence, b),
                    "edge labels": self.inference_edge_label(edge_labels, b),
                    "edge attributes": self.inference_edge_attribute(edge_attributes, b),
                    "id": batch["id"][b].cpu(),
                    "lemmas": batch["every_lemma"][b, : word_lens[b]].cpu(),
                    "tokens": batch["every_input"][0][b, : word_lens[b]].cpu(),
                    "token intervals": batch["token_intervals"][b, :, :].cpu(),
                },
                **kwargs
            )
            for b in range(batch_size)
        ]

        return outputs

    def loss(self, output, batch, matching, decoder_mask):
        batch_size = batch["every_input"][0].size(0)
        device = batch["every_input"][0].device
        T_label = batch["labels"][0].size(1)
        T_input = batch["every_input"][0].size(1)

        input_mask = create_padding_mask(batch_size, T_input, batch["every_input"][1], device)  # shape: (B, T_input)
        label_mask = create_padding_mask(batch_size, T_label, batch["labels"][1], device)  # shape: (B, T_label)
        edge_mask = torch.eye(T_label, T_label, device=device, dtype=torch.bool).unsqueeze(0)  # shape: (1, T_label, T_label)
        edge_mask = edge_mask | label_mask.unsqueeze(1) | label_mask.unsqueeze(2)  # shape: (B, T_label, T_label)
        edge_label_mask = (batch["edge_presence"] == 0) | edge_mask

        permute_edges(batch, output["edge presence"], edge_mask)
        if output["edge label"] is not None:
            batch["edge_labels"] = (
                batch["edge_labels"][0][:, :, :, :output["edge label"].size(-1)],
                batch["edge_labels"][1],
            )

        losses = {}
        losses.update(self.loss_label(output, batch, decoder_mask, matching))
        losses.update(self.loss_anchor(output, batch, input_mask, matching))
        losses.update(self.loss_edge_presence(output, batch, edge_mask))
        losses.update(self.loss_edge_label(output, batch, edge_label_mask.unsqueeze(-1)))
        losses.update(self.loss_edge_attribute(output, batch, edge_label_mask))
        losses.update(self.loss_property(output, batch, label_mask))
        losses.update(self.loss_top(output, batch, label_mask))

        stats = {f"{key} {self.language}-{self.framework}": value.detach().cpu().item() for key, value in losses.items()}
        total_loss = sum(losses[key] * self.loss_weights[key] for key in losses.keys())

        return total_loss, losses, stats

    @torch.no_grad()
    def create_cost_matrices(self, output, batch, decoder_lens):
        batch_size = len(batch["relative_labels"][1])
        decoder_lens = decoder_lens.cpu()

        matrices = []
        for b in range(batch_size):
            label_cost_matrix = self.label_cost_matrix(output, batch, decoder_lens, b)
            anchor_cost_matrix = self.anchor_cost_matrix(output, batch, decoder_lens, b)
            cost_matrix = label_cost_matrix * anchor_cost_matrix
            matrices.append(cost_matrix.cpu())

        return matrices

    def init_loss_weights(self, config):
        default_weight = 1.0 / len([v for v in config.values() if v])
        return nn.ParameterDict({k: nn.Parameter(torch.tensor([default_weight])) for k, v in config.items() if v})

    def init_edge_classifier(self, dataset, args, config, initialize: bool):
        if not config["edge presence"] and not config["edge label"] and not config["edge attribute"]:
            return None
        return EdgeClassifier(dataset, args, initialize, presence=config["edge presence"], label=config["edge label"], attribute=config["edge attribute"])

    def init_label_classifier(self, dataset, args, config, initialize: bool):
        if not config["label"]:
            return None
        return PaddingPacker(nn.Sequential(nn.Dropout(args.dropout_label), MixtureOfSoftmaxes(dataset, args, initialize)))

    def init_top_classifier(self, dataset, args, config, initialize: bool):
        if not config["top"]:
            return None
        return nn.Sequential(nn.Dropout(args.dropout_top), nn.Linear(args.hidden_size, 1, bias=False))

    def init_property_classifier(self, dataset, args, config, initialize: bool):
        if not config["property"]:
            return None

        classifier = nn.Sequential(nn.Dropout(args.dropout_property), nn.Linear(args.hidden_size, 1))

        if initialize:
            property_freq = dataset.property_freqs["transformed"][dataset.property_field.vocabs["transformed"].stoi[1]]
            classifier[1].bias.data.fill_((property_freq / (1.0 - property_freq)).log())

        return classifier

    def init_anchor_classifier(self, dataset, args, config, initialize: bool):
        if not config["anchor"]:
            return None
        return AnchorClassifier(dataset, args, initialize)

    def forward_edge(self, decoder_output):
        if self.edge_classifier is None:
            return None
        return self.edge_classifier(decoder_output, self.loss_weights)

    def forward_label(self, decoder_output, decoder_lens):
        if self.label_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["label"])
        return self.label_classifier(decoder_output, decoder_lens, decoder_output.size(1))

    def forward_top(self, decoder_output):
        if self.top_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["top"])
        return self.top_classifier(decoder_output).squeeze(-1)

    def forward_property(self, decoder_output):
        if self.property_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["property"])
        return self.property_classifier(decoder_output).squeeze(-1)

    def forward_anchor(self, decoder_output, encoder_output, encoder_mask):
        if self.anchor_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["anchor"])
        return self.anchor_classifier(decoder_output, encoder_output, encoder_mask)

    def inference_label(self, prediction):
        min_diff = (prediction[:, 0] - prediction[:, 1:].max(-1)[0]).min()
        if min_diff >= 0:
            prediction[:, 0] -= min_diff + 1e-3  # make sure at least one item will be selected

        return prediction.argmax(dim=-1)

    def inference_anchor(self, prediction):
        return prediction.sigmoid()

    def inference_property(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :].sigmoid().cpu()

    def inference_top(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :].cpu()

    def inference_edge_presence(self, prediction, example_index: int):
        if prediction is None:
            return None

        N = prediction.size(1)
        mask = torch.eye(N, N, device=prediction.device, dtype=torch.bool)
        return prediction[example_index, :, :].sigmoid().masked_fill(mask, 0.0).cpu()

    def inference_edge_label(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :, :, :].argmax(dim=-1).cpu()

    def inference_edge_attribute(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :, :, :].argmax(dim=-1).cpu()

    def loss_edge_presence(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge presence"] is None:
            return {}
        return {"edge presence": binary_cross_entropy(prediction["edge presence"], target["edge_presence"].float(), mask)}

    def loss_edge_label(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge label"] is None:
            return {}
        return {"edge label": binary_cross_entropy(prediction["edge label"], target["edge_labels"][0].float(), mask)}

    def loss_edge_attribute(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge attribute"] is None:
            return {}

        prediction = F.log_softmax(prediction["edge attribute"], dim=-1)
        return {"edge attribute": cross_entropy(prediction, target["edge_attributes"], mask)}

    def loss_label(self, prediction, target, mask, matching):
        if self.label_classifier is None or prediction["label"] is None:
            return {}

        prediction = prediction["label"]
        target, label_weight = match_smoothed_label(
            target["relative_labels"][0], matching, self.label_smoothing, prediction.shape, prediction.device, self.query_length, self.blank_weight
        )
        return {"label": multi_label_cross_entropy(prediction, target, mask, focal=self.focal, label_weight=label_weight)}

    def loss_top(self, prediction, target, mask):
        if self.top_classifier is None or prediction["top"] is None:
            return {}

        prediction = torch.log_softmax(prediction["top"].masked_fill(mask, float("-inf")), dim=-1)
        return {"top": cross_entropy(prediction, target["top"], None)}

    def loss_property(self, prediction, target, mask):
        if self.property_classifier is None or prediction["property"] is None:
            return {}
        return {"property": binary_cross_entropy(prediction["property"], target["properties"][:, :, 0].float(), mask)}

    def loss_anchor(self, prediction, target, mask, matching):
        if self.anchor_classifier is None or prediction["anchor"] is None:
            return {}

        prediction = prediction["anchor"]
        target, anchor_mask = match_anchor(target["anchor"], matching, prediction.shape, prediction.device)
        mask = anchor_mask.unsqueeze(-1) | mask.unsqueeze(-2)
        return {"anchor": binary_cross_entropy(prediction, target.float(), mask)}

    def label_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["label"] is None:
            return 1.0

        target_labels, _ = batch["relative_labels"]

        label_prob = output["label"][b, : decoder_lens[b], :].unsqueeze(0)  # shape: (1, num_queries, num_classes)
        tgt_label = (target_labels[b] > self.label_smoothing).long()  # shape: (num_nodes, num_inputs, num_classes)
        tgt_label = tgt_label.repeat_interleave(self.query_length, dim=1)  # shape: (num_nodes, num_queries, num_classes)
        cost_matrix = (tgt_label * label_prob).sum(-1).t()  # shape: (num_queries, num_nodes)
        return cost_matrix

    def anchor_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["anchor"] is None:
            return 1.0

        num_nodes = batch["relative_labels"][0][b].size(0)
        word_lens = batch["every_input"][1]
        target_anchors, _ = batch["anchor"]
        pred_anchors = output["anchor"].sigmoid()

        tgt_align = target_anchors[b, : num_nodes, : word_lens[b]]  # shape: (num_nodes, num_inputs)
        align_prob = pred_anchors[b, : decoder_lens[b], : word_lens[b]]  # shape: (num_queries, num_inputs)
        align_prob = align_prob.unsqueeze(1).expand(-1, num_nodes, -1)  # shape: (num_queries, num_nodes, num_inputs)
        align_prob = torch.where(tgt_align.unsqueeze(0).bool(), align_prob, 1.0 - align_prob)  # shape: (num_queries, num_nodes, num_inputs)
        cost_matrix = align_prob.log().mean(-1).exp()  # shape: (num_queries, num_nodes)
        return cost_matrix

    def loss_weights_dict(self):
        loss_weights = {f"{key} weight {self.language}-{self.framework}": weight.detach().cpu().item() for key, weight in self.loss_weights.items()}
        return loss_weights
