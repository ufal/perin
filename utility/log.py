#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from utility.loading_bar import LoadingBar
import time
import os
import json
import torch


class Log:
    def __init__(self, dataset, model, optimizer, args, directory, log_each: int, initial_epoch=-1, log_wandb=True):
        self.dataset = dataset
        self.model = model
        self.args = args
        self.optimizer = optimizer

        self.loading_bar = LoadingBar(length=27)
        self.best_f1_score = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_wandb = log_wandb
        if self.log_wandb:
            globals()["wandb"] = __import__("wandb")  # ugly way to not require wandb if not needed

        self.directory = directory
        self.evaluation_results = f"{directory}/results_{{0}}_{{1}}.json"
        self.full_evaluation_results = f"{directory}/full_results_{{0}}_{{1}}.json"
        self.best_full_evaluation_results = f"{directory}/best_full_results_{{0}}_{{1}}.json"
        self.result_history = {epoch: {} for epoch in range(args.epochs)}
        self.n_frameworks = len(args.frameworks)

        self.best_checkpoint_filename = f"{self.directory}/best_checkpoint.h5"
        self.last_checkpoint_filename = f"{self.directory}/last_checkpoint.h5"

        self.step = 0
        self.total_batch_size = 0
        self.flushed = True

    def train(self, len_dataset: int) -> None:
        self.flush()

        self.epoch += 1
        if self.epoch == 0:
            self._print_header()

        self.is_train = True
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, batch_size, losses, frameworks, grad_norm: float = None, learning_rates: float = None,) -> None:
        if self.is_train:
            self._train_step(batch_size, losses, grad_norm, learning_rates)
        else:
            self._eval_step(batch_size, losses)

        self._check_for_evaluation(frameworks)
        self.flushed = False

    def flush(self) -> None:
        if self.flushed:
            return
        self.flushed = True

        if self.is_train:
            print(f"\r┃{self.epoch:12d}  ┃{self._time():>12}  ┃", end="", flush=True)
        else:
            if self.losses is not None and self.log_wandb:
                dictionary = {f"validation {key}": value / self.step for key, value in self.losses.items()}
                dictionary["epoch"] = self.epoch
                wandb.log(dictionary)

            self.losses = None
            # self._save_model(save_as_best=False, performance=None)

    def _check_for_evaluation(self, frameworks):
        for framework, language in frameworks:
            evaluation_results = self.evaluation_results.format(framework, language)
            full_evaluation_results = self.full_evaluation_results.format(framework, language)

            if not os.path.exists(evaluation_results):
                continue

            try:
                with open(evaluation_results, mode="r") as f:
                    results = json.loads(f.readline())
                if "epoch" in results:
                    epoch = results["epoch"]
                    wandb.save(full_evaluation_results)
                    wandb.log(results)
                    os.remove(evaluation_results)
                else:
                    continue
            except:
                continue

            try:
                with open(full_evaluation_results, mode="r") as f:
                    results = json.loads(f.readline())
            except:
                continue

            self.result_history[epoch][(framework, language)] = results

            if len(self.result_history[epoch]) == self.n_frameworks:
                keys = ["tops", "labels", "properties", "anchors", "edges", "attributes", "all"]
                f = {key: 0.0 for key in keys}
                total = {key: 0 for key in keys}

                for result in self.result_history[epoch].values():
                    for key in keys:
                        f[key] += result[key]["g"] * result[key]["f"]
                        total[key] += result[key]["g"]
                del self.result_history[epoch]

                results = {f"evaluation {key} f1": f[key] / max(total[key], 1) for key in keys}
                results["epoch"] = epoch
                wandb.log(results)

                f1_score = results[f"evaluation all f1"]
                if f1_score > self.best_f1_score:
                    if self.log_wandb:
                        wandb.run.summary["best f1 score"] = f1_score
                    self.best_f1_score = f1_score
                    self._save_model(save_as_best=True, performance=results)

    def _save_model(self, save_as_best: bool, performance: dict):
        if not self.args.save_checkpoints:
            return

        state = {
            "epoch": self.epoch,
            "dataset": self.dataset.state_dict(),
            "performance": performance,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": self.args.state_dict(),
        }

        filename = self.best_checkpoint_filename if save_as_best else self.last_checkpoint_filename

        torch.save(state, filename)
        if self.log_wandb:
            wandb.save(filename)

    def _train_step(self, batch_size, losses, grad_norm: float, learning_rates) -> None:
        self.total_batch_size += batch_size
        self.step += 1

        if self.losses is None:
            self.losses = losses
        else:
            for key, values in losses.items():
                if key not in self.losses:
                    self.losses[key] = losses[key]
                    continue
                self.losses[key] += losses[key]

        if self.step % self.log_each == 0:
            progress = self.total_batch_size / self.len_dataset
            print(f"\r┃{self.epoch:12d}  │{self._time():>12}  {self.loading_bar(progress)}", end="", flush=True)

            if self.log_wandb:
                dictionary = {f"train {key}": value / self.log_each for key, value in self.losses.items()}
                dictionary["epoch"] = self.epoch
                dictionary["learning rate - encoder"] = learning_rates[-3]
                dictionary["learning rate - decoder"] = learning_rates[-2]
                dictionary["learning rate - grad_norm"] = learning_rates[-1]
                dictionary["gradient norm"] = grad_norm

                wandb.log(dictionary)

            self.losses = None

    def _eval_step(self, batch_size, losses) -> None:
        self.step += 1

        if self.losses is None:
            self.losses = losses
        else:
            for key, values in losses.items():
                if key not in self.losses:
                    self.losses[key] = losses[key]
                    continue
                self.losses[key] += losses[key]

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.total_batch_size = 0
        self.len_dataset = len_dataset
        self.losses = None

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━╸S╺╸E╺╸M╺╸A╺╸N╺╸T╺╸I╺╸S╺╸K╺━━━━━━━━━━━━━━┓")
        print(f"┃              ┃              ╷                             ┃")
        print(f"┃       epoch  ┃     elapsed  │               progress bar  ┃")
        print(f"┠──────────────╂──────────────┼─────────────────────────────┨")
