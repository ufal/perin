#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from utility.label_processor import LabelProcessor

processor = LabelProcessor()


def check(forms, lemma, label, rule_classes, concat: bool, num_separator=''):
    rules = list(processor.gen_all_label_rules(forms, forms, label, rule_classes, ['', '+', '-'], concat, num_separator=num_separator))

    for rule in rules:
        assert processor.apply_label_rule(forms, forms, rule, concat) == label

    return rules


def test_all_1():
    forms = ["hello"]
    label = "hello"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas"]
    rules = check(forms, forms, label, rule_classes, False)

    assert len(rules) >= 3
    assert "a│hello" in rules
    assert "d││0,0││" in rules
    assert "l││0,0││" in rules


def test_all_2():
    forms = ["said"]
    label = "say-01"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas"]
    rules = check(forms, forms, label, rule_classes, False)

    assert len(rules) >= 3
    assert "a│say-01" in rules
    assert "d││0,0││--+y+-+0+1" in rules
    assert "l││0,0││--+y+-+0+1" in rules


def test_all_3():
    forms = ["presaid"]
    label = "say-01"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas"]
    rules = check(forms, forms, label, rule_classes, False)

    assert len(rules) >= 3
    assert "a│say-01" in rules
    assert "d││0,0│---│--+y+-+0+1" in rules
    assert "l││0,0│---│--+y+-+0+1" in rules


def test_number_1():
    forms = ["3", "thousand", "five", "hundred"]
    label = "3500"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas", "numerical_divide"]
    rules = check(forms, forms, label, rule_classes, False)

    assert len(rules) >= 4
    assert "a│3500" in rules
    assert "n│4│" in rules
    assert "d││0,0││+5+0+0" in rules
    assert "l││0,0││+5+0+0" in rules


def test_number_2():
    forms = ["3", "thousand", "five", "hundred"]
    label = "3_500"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas", "numerical_all"]
    rules = check(forms, forms, label, rule_classes, False, num_separator='_')

    assert len(rules) >= 4
    assert "a│3_500" in rules
    assert "n│-1│_" in rules
    assert "d││0,0││+_+5+0+0" in rules
    assert "l││0,0││+_+5+0+0" in rules


def test_concat_1():
    forms = ["all", "-", "in"]
    label = "all-in"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas", "concatenate"]
    rules = check(forms, forms, label, rule_classes, False)

    assert len(rules) >= 4
    assert "a│all-in" in rules
    assert "c│3│││" in rules
    assert "d││0,0││+-+i+n" in rules
    assert "l││0,0││+-+i+n" in rules


def test_concat_2():
    forms = ["such", "as"]
    label = "_such+as_p"
    rule_classes = ["absolute", "relative_forms", "relative_lemmas", "concatenate"]
    rules = check(forms, forms, label, rule_classes, False)

    assert len(rules) >= 4
    assert "a│_such+as_p" in rules
    assert "c│2│+│_│_p" in rules
    assert "d││0,0│+_│+++a+s+_+p" in rules
    assert "l││0,0│+_│+++a+s+_+p" in rules


def test_concat_3():
    forms = ["all", "-", "in"]
    label = "all-in"
    rule_classes = ["relative_forms", "relative_lemmas"]
    rules = check(forms, forms, label, rule_classes, True)

    assert len(rules) >= 2
    assert "d││0,0││" in rules
    assert "l││0,0││" in rules


def test_concat_4():
    forms = ["such", "as"]
    label = "_such+as_p"
    rule_classes = ["relative_forms", "relative_lemmas"]
    rules = check(forms, forms, label, rule_classes, True)

    assert len(rules) >= 2
    assert "d│+│0,0│+_│+_+p" in rules
    assert "l│+│0,0│+_│+_+p" in rules
