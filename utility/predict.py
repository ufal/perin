#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json
import torch

from data.batch import Batch
from utility.evaluate import evaluate


def sentence_condition(s, f, l):
    return ("framework" not in s or f == s["framework"]) and ("framework" in s or f in s["targets"])


def predict(model, data, input_paths, args, output_directory, gpu, run_evaluation=False, epoch=None):
    model.eval()
    input_files = {(f, l): input_paths[(f, l)] for f, l in args.frameworks}

    sentences = {(f, l): {} for f, l in args.frameworks}
    for framework, language in args.frameworks:
        with open(input_files[(framework, language)], encoding="utf8") as f:
            for line in f.readlines():
                line = json.loads(line)

                if not sentence_condition(line, framework, language):
                    continue

                line["nodes"] = []
                line["edges"] = []
                line["tops"] = []
                line["framework"] = framework
                line["language"] = language
                sentences[(framework, language)][line["id"]] = line

    for i, batch in enumerate(data):
        with torch.no_grad():
            all_predictions = model(Batch.to(batch, gpu), inference=True)

        for (framework, language), predictions in all_predictions.items():
            for prediction in predictions:
                for key, value in prediction.items():
                    sentences[(framework, language)][prediction["id"]][key] = value

    for framework, language in args.frameworks:
        output_path = f"{output_directory}/prediction_{framework}_{language}.json"
        with open(output_path, "w", encoding="utf8") as f:
            for sentence in sentences[(framework, language)].values():
                json.dump(sentence, f, ensure_ascii=False)
                f.write("\n")
                f.flush()

        if args.log_wandb:
            import wandb
            wandb.save(output_path)

        if run_evaluation:
            # this should be run in parallel, if your setup allows it
            evaluate(output_directory, epoch, framework, language, input_files[(framework, language)])
