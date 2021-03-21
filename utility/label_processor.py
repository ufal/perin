#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from utility.num_converter import NumConverter


class LabelProcessor:
    def __init__(self):
        self.converter = NumConverter()

    def is_absolute_label_rule(self, label_rule):
        return label_rule.startswith("a")

    def make_absolute_label_rule(self, label):
        return f"a│{label.lower()}"

    def gen_all_label_rules(self, forms, lemmas, label, rule_classes, separators, concat=False,
                            allow_copy=False, num_separator='', num_lemmas=False, ignore_nonalnum=False):
        label = label.lower()

        if ignore_nonalnum:
            forms = self.filter_nonalnum(forms)
            lemmas = self.filter_nonalnum(lemmas)

        if "absolute" in rule_classes:
            yield self.make_absolute_label_rule(label)
        if "relative_forms" in rule_classes:
            yield from self.gen_all_relative_rules(forms, label, separators, allow_copy=allow_copy, concat=concat, diff_rule='d')
        if "relative_lemmas" in rule_classes:
            yield from self.gen_all_relative_rules(lemmas, label, separators, allow_copy=allow_copy, concat=concat, diff_rule='l')
        if "numerical_divide" in rule_classes:
            yield from self.gen_all_numerical_rules(lemmas if num_lemmas else forms, label, divide=True, num_separator=num_separator)
        if "numerical_all" in rule_classes:
            yield from self.gen_all_numerical_rules(lemmas if num_lemmas else forms, label, divide=False, num_separator=num_separator)
        if "concatenate" in rule_classes:
            yield from self.gen_all_concatenation_rules(forms, label, separators)

    def filter_nonalnum(self, tokens):
        if len(tokens) == 1:
            return tokens

        if len(tokens[0]) == 0 or (not tokens[0][0].isalnum() and tokens[0][0] != '-'):
            if len(tokens[0]) > 1:
                tokens[0] = tokens[0][1:]
            else:
                tokens = tokens[1:]
        if len(tokens[-1]) == 0 or not tokens[-1][-1].isalnum():
            if len(tokens[-1]) > 1:
                tokens[-1] = tokens[-1][:-1]
            else:
                tokens = tokens[:-1]

        if len(tokens) == 0:
            return [""]
        return tokens

    def gen_all_numerical_rules(self, forms, label, divide: bool, num_separator=''):
        if self.converter.is_number(label, num_separator):
            if divide:
                for i, result in enumerate(self.converter.to_all_numbers(forms, num_separator)):
                    if result is not None and result == label:
                        yield f"n│{i+1}│{num_separator}"
            else:
                result = self.converter.to_number(forms, num_separator)
                if result is not None and result == label:
                    yield f"n│-1│{num_separator}"

    def gen_all_concatenation_rules(self, forms, label, separators):
        if len(forms) > 1:
            length = len(forms[0])

            for i in range(1, len(forms)):
                length += len(forms[i])
                if length > len(label):
                    break

                for separator in separators:
                    processed = separator.join(forms[:i + 1]).lower()
                    index = label.find(processed)

                    if index >= 0:
                        yield f"c│{i + 1}│{separator}│{label[:index]}│{label[index+len(processed):]}"

    def gen_all_relative_rules(self, forms, label, separators, allow_copy: bool, concat: bool, diff_rule: str):
        if concat and len(forms) > 1:
            for separator in separators:
                for pre in range(len(forms)):
                    for suf in range(len(forms) - pre):
                        form = separator.join(forms[pre:len(forms) - suf]).lower()
                        join_rule = f"{separator}│{pre},{suf}"
                        yield from self._gen_all_relative_rules(form, label, allow_copy, diff_rule, join_rule)
        else:
            yield from self._gen_all_relative_rules(forms[0].lower(), label, allow_copy, diff_rule, f"│0,0")

    def _gen_all_relative_rules(self, form, label, allow_copy: bool, diff_rule: str, separator: str):
        for l in range(len(label)):
            for f in range(len(form)):
                cpl = 0
                while f + cpl < len(form) and l + cpl < len(label) and form[f + cpl] == label[l + cpl]:
                    cpl += 1
                    # if cpl == 1 and len(form) > 3 and len(label) > 3:
                    #     continue

                    prefix = self.min_edit_script(form[:f], label[:l], allow_copy)
                    suffix = self.min_edit_script(form[f + cpl :], label[l + cpl :], allow_copy)

                    yield f"{diff_rule}│{separator}│{prefix}│{suffix}"

    def apply_label_rule(self, forms, lemmas, rule, concat: bool, num_lemmas=False, ignore_nonalnum=False):
        if ignore_nonalnum:
            forms = self.filter_nonalnum(forms)
            lemmas = self.filter_nonalnum(lemmas)

        form = forms[0].lower()
        processor, rule = rule[0], rule[2:]

        if processor == "a":
            return rule

        if processor == "n":
            parts = rule.split('│')
            n_items = int(parts[0])
            n_items = len(forms) if n_items == -1 else n_items
            separator = parts[1] if len(parts) > 1 else ''

            return self.converter.to_number(lemmas[:n_items] if num_lemmas else forms[:n_items], separator)

        if processor == "c":
            n_items, separator, prefix, suffix = rule.split('│')
            return prefix + separator.join(forms[:int(n_items)]).lower() + suffix

        if processor == "l":
            forms = lemmas
        else:
            assert processor == "d"

        separator, remover, *rules = rule.split('│')
        prefix, suffix = map(int, remover.split(','))
        form = separator.join(forms[prefix : len(forms) - suffix]).lower() if concat else forms[0].lower()
        rule_sources = []
        assert len(rules) == 2, f"{rule}"

        for rule in rules:
            source, i = 0, 0
            while i < len(rule):
                if rule[i] == "→" or rule[i] == "-":
                    source += 1
                else:
                    assert rule[i] == "+"
                    i += 1
                i += 1
            rule_sources.append(source)

        try:
            label = ""
            for i in range(2):
                j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                while j < len(rules[i]):
                    if rules[i][j] == "→":
                        label += form[offset]
                        offset += 1
                    elif rules[i][j] == "-":
                        offset += 1
                    else:
                        assert rules[i][j] == "+"
                        label += rules[i][j + 1]
                        j += 1
                    j += 1
                if i == 0:
                    label += form[rule_sources[0] : len(form) - rule_sources[1]]
        except:
            label = form

        return label

    def min_edit_script(self, source, target, allow_copy=False):
        if allow_copy:
            return self.min_edit_script_copy(source, target)
        if len(target) == 0:
            return '-' * len(source)
        return '-' * len(source) + '+' + '+'.join(target)

    def min_edit_script_copy(self, source, target):
        a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
        for i in range(0, len(source) + 1):
            for j in range(0, len(target) + 1):
                if i == 0 and j == 0:
                    a[i][j] = (0, "")
                else:
                    if i and j and source[i - 1] == target[j - 1] and a[i - 1][j - 1][0] < a[i][j][0]:
                        a[i][j] = (a[i - 1][j - 1][0], a[i - 1][j - 1][1] + "→")
                    if i and a[i - 1][j][0] < a[i][j][0]:
                        a[i][j] = (a[i - 1][j][0] + 1, a[i - 1][j][1] + "-")
                    if j and a[i][j - 1][0] < a[i][j][0]:
                        a[i][j] = (a[i][j - 1][0] + 1, a[i][j - 1][1] + "+" + target[j - 1])
        return a[-1][-1][1]
