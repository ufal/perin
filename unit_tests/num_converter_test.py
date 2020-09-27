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


converter = NumConverter()


def test_to_all():
    options = list(converter.to_all_numbers(["6", "hundred", "million", "GPUs"]))
    assert options[0] == "6"
    assert options[1] == "600"
    assert options[2] == "600000000"
    assert options[3] is None


def test_simple_num():
    assert converter.to_number(["1"]) == "1"


def test_simple_word_1():
    assert converter.to_number(["one"]) == "1"


def test_simple_word_2():
    assert converter.to_number(["seventy"]) == "70"


def test_simple_word_3():
    assert converter.to_number(["seventeen"]) == "17"


def test_simple_word_4():
    assert converter.to_number(["first"]) == "1"


def test_simple_word_5():
    assert converter.to_number(["second"]) == "2"


def test_simple_word_6():
    assert converter.to_number(["eleventh"]) == "11"


def test_simple_hybrid_1():
    assert converter.to_number(["1st"]) == "1"


def test_simple_hybrid_2():
    assert converter.to_number(["22nd"]) == "22"


def test_complex_num_1():
    assert converter.to_number(["1", ",", "000"]) == "1000"


def test_complex_num_2():
    assert converter.to_number(["1", ",", "234", ".", "567"]) == "1234.567"


def test_complex_num_3():
    assert converter.to_number(["-", "1", ",", "234", ".", "567"]) == "-1234.567"


def test_complex_word():
    assert converter.to_number(["seven", "millions"]) == "7000000"


def test_complex_word_2():
    assert converter.to_number(["seven", "million"]) == "7000000"


def test_complex_word_3():
    assert converter.to_number(["seventy", "three"]) == "73"


def test_complex_word_4():
    assert converter.to_number(["seven", "million", "five", "thousand", "four", "hundred", "and", "seventy", "five"]) == "7005475"


def test_complex_hybrid_1():
    assert converter.to_number(["7", "million"]) == "7000000"


def test_complex_hybrid_2():
    assert converter.to_number(["seven", "million", "5", "thousand", "four", "hundred", "and", "75"]) == "7005475"


def test_complex_hybrid_3():
    assert converter.to_number(["1", ",", "000", "thousand"]) == "1000000"


def test_complex_hybrid_4():
    assert converter.to_number(["0", ".", "25", "million"]) == "250000"


def test_complex_hybrid_5():
    assert converter.to_number(["0", ".", "2", "million"]) == "200000"


def test_complex_hybrid_6():
    assert converter.to_number([".", "2", "million"]) == "200000"


def test_complex_hybrid_7():
    assert converter.to_number(["1", ",", "234", ".", "567", "million"]) == "1234567000"


def test_complex_hybrid_8():
    assert converter.to_number(["100", "billion"]) == "100000000000"


def test_complex_hybrid_9():
    assert converter.to_number(["100", "-", "billion"]) == "100000000000"


def test_complex_hybrid_10():
    assert converter.to_number(["10s", "of", "millions"]) == "10000000"


def test_chinese_1():
    assert converter.to_number(["五十八"]) == "58"


def test_chinese_2():
    assert converter.to_number(["一千一百八十六亿"]) == "118600000000"


def test_chinese_3():
    assert converter.to_number(["五十八点一"]) == "58.1"


def test_month_1():
    assert converter.to_number(["january"]) == "1"


def test_month_2():
    assert converter.to_number(["February"]) == "2"


def test_month_3():
    assert converter.to_number(["Dec"]) == "12"


def test_fail_1():
    assert converter.to_number(["hello"]) is None


def test_fail_2():
    assert converter.to_number(["1hello"]) is None


def test_fail_3():
    assert converter.to_number([""]) is None


def test_fail_5():
    assert converter.to_number(["1", ".", "234", ".", "567"]) is None


def test_fail_6():
    assert converter.to_number(["."]) is None


def test_fail_7():
    assert converter.to_number([","]) is None


def test_fail_8():
    assert converter.to_number([","]) is None


def test_fail_9():
    assert converter.to_number(["国际"]) is None
