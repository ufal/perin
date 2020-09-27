#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

def bert_tokenizer(example, tokenizer, encoder):
    if "xlm" in encoder.lower():
        separator = '▁'
    elif "roberta" in encoder.lower():
        separator = 'Ġ'
    elif "bert" in encoder.lower():
        separator = '##'
    else:
        raise Exception(f"Unsupported tokenization for {encoder}")

    sentence = " ".join(example["input"]).replace("  ", " ").replace("´", "'")
    original_tokens = [''.join([t.lstrip(separator).lower().strip() for t in tokenizer.tokenize(token)]) for token in example["input"]]
    tokenized_tokens = [token.lstrip(separator).lower().strip() for token in tokenizer.tokenize(sentence)]

    to_scatter, to_gather, to_delete = [], [], []
    orig_i, orig_offset, chain_length = 0, 0, 0
    unk_roll = False

    for i, token in enumerate(tokenized_tokens):
        chain_length += 1

        while orig_i < len(original_tokens) - 1 and orig_offset >= len(original_tokens[orig_i]):
            orig_i, orig_offset = orig_i + 1, 0
            chain_length = 0

        if token == "[unk]":
            unk_roll = True
            to_gather.append(i + 1)
            to_scatter.append(orig_i)
            if chain_length > 5:
                to_delete.append(i)
            continue

        elif unk_roll:
            found = False
            for orig_i in range(orig_i, len(original_tokens)):
                for orig_offset in range(len(original_tokens[orig_i])):
                    original_token = original_tokens[orig_i][orig_offset:]
                    if original_token.startswith(token) or token.startswith(original_token):
                        chain_length = 0
                        found = True
                        break
                if found:
                    break

        original_token = original_tokens[orig_i][orig_offset:]
        unk_roll = False

        if original_token.startswith(token):
            to_gather.append(i + 1)
            to_scatter.append(orig_i)
            orig_offset += len(token)
            if chain_length > 5:
                to_delete.append(i)
            continue

        print(f"BERT parsing error in sentence {example['id']}: {example['sentence']}")

        unk_roll = True
        to_gather.append(i + 1)
        to_scatter.append(orig_i)

    bert_input = tokenizer.encode(sentence, add_special_tokens=True)
    to_gather, to_scatter, bert_input = reduce_bert_input(to_gather, to_scatter, bert_input, to_delete)

    return to_scatter, bert_input


def reduce_bert_input(to_gather, to_scatter, bert_input, to_delete):
    new_gather, new_scatter = [], []
    offset = 0
    for i in range(len(to_gather)):
        if to_gather[i] - 1 in to_delete:
            offset += 1
        else:
            new_gather.append(to_gather[i] - offset)
            new_scatter.append(to_scatter[i])
    bert_input = [w for i, w in enumerate(bert_input) if i - 1 not in to_delete]
    return new_gather, new_scatter, bert_input
