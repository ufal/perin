#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import json
import torch

from PIL import Image
from subprocess import run

from data.batch import Batch
from utility.evaluate import evaluate
from utility.utils import resize_to_square


def sentence_condition(s, f, l):
    return ("framework" not in s or f == s["framework"]) and ("framework" in s or f in s["targets"])


def predict(model, data, input_paths, args, output_directory, gpu, eval_script=None, visual_script=None, epoch=None):
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

        # if visual_script is not None:
        #     image_name = f"sample_{framework}_{language}"
        #     n_samples = 6
        #     run([visual_script, str(n_samples), output_path, f"{output_directory}/{image_name}", framework, input_files[(framework, language)]], cwd=os.getcwd())

        #     images = [Image.open(f"{output_directory}/{image_name}{i}.png") for i in range(n_samples)]
        #     images = [wandb.Image(resize_to_square(image, 1024)) for image in images]

        #     wandb.log({f"{framework}-{language} prediction #{i}": image for i, image in enumerate(images)})

        if eval_script is not None:
            result = run(
                [
                    "qsub",
                    "-N",
                    "EVALUERING",
                    "-cwd",
                    "-pe",
                    "smp",
                    "1",
                    "-l",
                    "mem_free=32G,act_mem_free=32G,h_data=32G",
                    "-q",
                    "cpu*",
                    "-j",
                    "y",
                    "-now",
                    "y",
                    eval_script,
                    output_directory,
                    str(epoch),
                    framework,
                    language,
                    input_files[(framework, language)],
                ],
                cwd=os.getcwd(),
            )
            if result.returncode != 0:
                evaluate(output_directory, epoch, framework, language, input_files[(framework, language)])
