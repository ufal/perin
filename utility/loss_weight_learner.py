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
import torch.nn.functional as F
import torch.distributed as dist

from utility.adamw import AdamW
from utility.schedule.inverse_sqrt_lr import InverseSqrtLr


class LossWeightLearner:
    def __init__(self, args, model, n_gpus):
        self.all_loss_weights = [head.loss_weights for head in model.heads]
        self.all_loss_keys = [list(loss_weights.keys()) for loss_weights in self.all_loss_weights]
        self.optimizer = AdamW([l for ll in self.all_loss_weights for l in ll.values()], lr=args.grad_norm_lr, weight_decay=0)
        self.scheduler = InverseSqrtLr(self.optimizer.param_groups[0], args.grad_norm_lr, args.warmup_steps, args.encoder_delay_steps)
        self.losses_0 = [0.0 for _ in model.heads]
        self.last_layer = list(model.decoder.layers[-1].parameters())
        self.n_gpus = n_gpus
        self.alpha = args.grad_norm_alpha
        self.accumulation_steps = args.accumulation_steps
        self.accumulated_grads = 0.0
        self.distributed = args.distributed

    def compute_grad(self, all_losses, epoch: int):
        for loss_weights in self.all_loss_weights:
            loss_weights.zero_grad()

        all_grads = []
        for j, losses in enumerate(all_losses):
            if len(losses) == 0:
                grads = [torch.zeros_like(self.all_loss_weights[j][key]) for key in self.all_loss_keys[j]]
            else:
                grads = {}
                for name, loss in losses.items():
                    grads[name] = torch.cat([g.flatten() for g in torch.autograd.grad(loss, self.last_layer, retain_graph=True)])
                grads = torch.cat([
                    (torch.norm(grads[key], 2) / self.all_loss_weights[j][key]).detach() * self.all_loss_weights[j][key]
                    for key in self.all_loss_keys[j]
                ])

                losses = torch.stack([losses[key] for key in self.all_loss_keys[j]])

                if epoch == 0:
                    self.losses_0[j] = self.losses_0[j] * 0.9 + losses.detach() * 0.1

                with torch.no_grad():
                    target = losses / self.losses_0[j]
                    target.div_(target.mean())
                    target.pow_(self.alpha).mul_(grads.mean())

                grad_norm_loss = F.l1_loss(grads, target, reduction="sum")
                with torch.no_grad():
                    grads = torch.autograd.grad(grad_norm_loss, [self.all_loss_weights[j][key] for key in self.all_loss_keys[j]])

            all_grads.append(torch.cat(grads))

        self.accumulated_grads += torch.cat(all_grads)

    @torch.no_grad()
    def step(self, epoch: int):
        if self.distributed:
            dist.all_reduce(self.accumulated_grads)
            self.accumulated_grads.div_(self.n_gpus)

        offset = 0
        for j in range(len(self.all_loss_weights)):
            for key in self.all_loss_keys[j]:
                self.all_loss_weights[j][key].grad = self.accumulated_grads[offset].unsqueeze(0).clone()
                offset += 1
        self.accumulated_grads.zero_()

        self.scheduler(epoch)
        self.optimizer.step()

        normalize_coeff = 0.0
        for j, loss_weights in enumerate(self.all_loss_weights):
            normalize_coeff += torch.stack(list(loss_weights.values())).sum()

        normalize_coeff = len(self.all_loss_weights) / normalize_coeff
        for j, loss_weights in enumerate(self.all_loss_weights):
            # normalize_coeff = 1.0 / torch.stack(list(loss_weights.values())).sum()
            for key in self.all_loss_keys[j]:
                self.all_loss_weights[j][key].data = loss_weights[key].data * normalize_coeff
