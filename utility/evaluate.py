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
import json
import multiprocessing as mp

import mtool.main
import mtool.score.mces


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch.")
    parser.add_argument("--framework", type=str, default="", help="Framework.")
    parser.add_argument("--language", type=str, default="", help="Language.")
    parser.add_argument("--input_dir", type=str, default="", help="Path to the run directory.")
    parser.add_argument("--gold_file", type=str, default="", help="Path to the gold file.")
    return parser.parse_args()


def evaluate(input_dir, epoch, framework, language, gold_file):
    normalize = {"anchors", "case", "edges", "attributes"}
    cores = mp.cpu_count()

    with open(f"{input_dir}/prediction_{framework}_{language}.json", encoding="utf8") as f:
        graphs, _ = mtool.main.read_graphs(f, format="mrp", frameworks=[framework], normalize=normalize)
        for graph in graphs:
            graph._language = None

    with open(gold_file, encoding="utf8") as f:
        gold_graphs, _ = mtool.main.read_graphs(f, format="mrp", frameworks=[framework], normalize=normalize)
        for graph in gold_graphs:
            graph._language = None

    limits = {"rrhc": 2, "mces": 50000}
    result = mtool.score.mces.evaluate(gold_graphs, graphs, limits=limits, cores=cores)

    with open(f"{input_dir}/full_results_{framework}_{language}.json", mode="w") as f:
        json.dump(result, f)

    result = {
        "epoch": epoch,
        f"evaluation tops accuracy {framework}-{language}": result["tops"]["f"],
        f"evaluation anchors f1 {framework}-{language}": result["anchors"]["f"],
        f"evaluation labels f1 {framework}-{language}": result["labels"]["f"],
        f"evaluation properties f1 {framework}-{language}": result["properties"]["f"],
        f"evaluation edges f1 {framework}-{language}": result["edges"]["f"],
        f"evaluation attributes f1 {framework}-{language}": result["attributes"]["f"],
        f"evaluation all f1 {framework}-{language}": result["all"]["f"],
    }

    with open(f"{input_dir}/results_{framework}_{language}.json", mode="w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.input_dir, args.epoch, args.framework, args.language, args.gold_file)
