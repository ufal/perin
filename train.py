#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from transformers import AutoConfig

from model.model import Model
from data.shared_dataset import SharedDataset
from utility.initialize import initialize
from utility.log import Log
from utility.schedule.multi_scheduler import multi_scheduler_wrapper
from utility.autoclip import AutoClip
from data.batch import Batch
from config.params import Params
from utility.predict import predict
from utility.adamw import AdamW
from utility.loss_weight_learner import LossWeightLearner


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--data_directory", type=str, default="/home/samueld/mrp_update/mrp")
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--dist_url", default="localhost", type=str)
    parser.add_argument("--log_wandb", dest="log_wandb", action="store_true", default=False)
    parser.add_argument("--name", default="default", type=str, help="name of this run.")
    parser.add_argument("--save_checkpoints", dest="save_checkpoints", action="store_true", default=False)
    parser.add_argument("--wandb_log_mode", type=str, default=None, help="How to log the model weights, supported values: {'all', 'gradients', 'parameters', None}")
    parser.add_argument("--workers", type=int, default=2, help="number of CPU workers per GPU.")
    args = parser.parse_args()

    params = Params()
    params.load(args)
    params.load_state_dict(vars(args))

    encoder_config = AutoConfig.from_pretrained(params.encoder)
    params.hidden_size = encoder_config.hidden_size
    params.n_encoder_layers = encoder_config.num_hidden_layers

    return params


def main_worker(gpu, n_gpus_per_node, args):
    is_master = gpu == 0
    directory = initialize(args, create_directory=is_master, init_wandb=args.log_wandb and is_master)

    os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://", world_size=n_gpus_per_node, rank=gpu)

    dataset = SharedDataset(args)
    dataset.load_datasets(args, gpu, n_gpus_per_node)

    model = Model(dataset, args)
    parameters = [{"params": p, "weight_decay": args.encoder_weight_decay} for p in model.get_encoder_parameters(args.n_encoder_layers)] + [
        {"params": model.get_decoder_parameters(), "weight_decay": args.decoder_weight_decay}
    ]
    optimizer = AdamW(parameters, betas=(0.9, args.beta_2))
    scheduler = multi_scheduler_wrapper(optimizer, args)
    autoclip = AutoClip([p for name, p in model.named_parameters() if "loss_weights" not in name])
    if args.balance_loss_weights:
        loss_weight_learner = LossWeightLearner(args, model, n_gpus_per_node)

    if is_master:
        if args.log_wandb:
            import wandb
            wandb.watch(model, log=args.wandb_log_mode)
        print(f"\nmodel: {model}\n")
        log = Log(dataset, model, optimizer, args, directory, log_each=10, log_wandb=args.log_wandb)

    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        raw_model = model.module
    else:
        raw_model = model

    for epoch in range(args.epochs):

        #
        # TRAINING
        #

        model.train()
        if is_master:
            log.train(len_dataset=dataset.train_size)

        i = 0
        model.zero_grad()

        for batch in dataset.train:
            batch = Batch.to(batch, gpu)
            total_loss, losses, stats = model(batch)

            for head in raw_model.heads:
                stats.update(head.loss_weights_dict())

            if args.balance_loss_weights:
                loss_weight_learner.compute_grad(losses, epoch)
            total_loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                grad_norm = autoclip()

                if args.balance_loss_weights:
                    loss_weight_learner.step(epoch)
                scheduler(epoch)
                optimizer.step()
                model.zero_grad()

                if is_master:
                    with torch.no_grad():
                        batch_size = batch["every_input"][0].size(0) * args.accumulation_steps
                        log(batch_size, stats, args.frameworks, grad_norm=grad_norm, learning_rates=scheduler.lr() + [loss_weight_learner.scheduler.lr()])

            del total_loss, losses

            i += 1

        if not is_master:
            continue

        #
        # VALIDATION CROSS-ENTROPIES
        #
        model.eval()
        log.eval(len_dataset=dataset.val_size)

        with torch.no_grad():
            for batch in dataset.val:
                try:
                    _, _, stats = model(Batch.to(batch, gpu))

                    batch_size = batch["every_input"][0].size(0)
                    log(batch_size, stats, args.frameworks)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

        log.flush()

        #
        # VALIDATION MRP-SCORES
        #
        predict(raw_model, dataset.val, args.validation_data, args, directory, gpu, run_evaluation=True, epoch=epoch)

    #
    # TEST PREDICTION
    #
    os.mkdir(f"{directory}/test_predictions/")
    predict(raw_model, dataset.test, args.test_data, args, f"{directory}/test_predictions/", gpu)


if __name__ == "__main__":
    args = parse_arguments()

    n_gpu = torch.cuda.device_count()
    args.distributed = n_gpu > 1
    args.batch_size = args.batch_size // max(n_gpu, 1)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpu, args=(n_gpu, args), join=True)
        dist.destroy_process_group()
    else:
        main_worker(0, 1, args)
