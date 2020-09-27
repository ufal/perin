#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from text2digits import text2digits
from pycnnum import cn2num
from decimal import Decimal
import re


class NumConverter:
    def __init__(self):
        self.t2d = text2digits.Text2Digits()

    def to_all_numbers(self, words, separator=''):
        for i in range(len(words)):
            yield self.to_number(words[: i + 1], separator)

    def to_number(self, words, separator=''):
        if words[0] == "-":
            mult = -1
            words = words[1:]
        else:
            mult = 1

        words = list(self.cluster_words(words))
        if None in words:
            return None

        # text2digit cannot handle decimal points
        if any("." in w for w in words):
            number = Decimal(1)
            for word in words:
                word = word.lower()
                new = self._base_number(word)
                if isinstance(new, str):
                    new = self._text_to_digit(new)
                    new = self._direct_transform(new)
                if new is None:
                    return None

                number *= new
            return self._decimal_to_str(mult * number, separator)

        words = " ".join(words)
        number = self._text_to_digit(words)
        if number is None or not self.is_number(number):
            return None
        number = mult * self.to_decimal(number)
        return self._decimal_to_str(number, separator)

    def cluster_words(self, words):
        base = ""
        for word in words:
            word = word.lower()
            if self.is_number(word) or word in [",", "."]:
                base += word
            elif word in [" ", "-", "of"]:
                continue
            else:
                if len(base) > 0:
                    yield self._decimal_to_str(self._base_number(base))
                yield self._decimal_to_str(self._base_number(word))
                base = ""

        if len(base) > 0:
            yield self._decimal_to_str(self._base_number(base))

    def _decimal_to_str(self, number, separator=''):
        if isinstance(number, Decimal):
            if number.to_integral() == number:
                number = number.to_integral()
            return self.separate_thousands(number, separator)
        if isinstance(number, str):
            return number
        return self.separate_thousands(number, separator)

    def _text_to_digit(self, word):
        try:
            if word == '〇' or word == '零':
                return '0'
            num = cn2num(word)
            if num != 0:
                return str(num)
        except:
            pass

        try:
            return self.t2d.convert(word.lower())
        except:
            return None

    def _base_number(self, word):
        direct = self._direct_transform(word)
        if direct is not None:
            return direct

        if len(word) > 1 and word[-1] == "s":
            converted = self._text_to_digit(word[:-1])
            direct = self._direct_transform(converted)
            if direct is not None:
                return word[:-1]

        if len(word) > 2 and word[-2:] in ["st", "nd", "rd", "th"]:
            converted = self._text_to_digit(word[:-2])
            direct = self._direct_transform(converted)
            if direct is not None:
                return word[:-2]

        return word

    def _direct_transform(self, word):
        if word is None:
            return None

        if self.is_number(word):
            return self.to_decimal(word)

        if word in ["january", "jan"]:
            return 1
        if word in ["february", "feb"]:
            return 2
        if word in ["march", "mar"]:
            return 3
        if word in ["april", "apr"]:
            return 4
        if word in ["may"]:
            return 5
        if word in ["june", "jun"]:
            return 6
        if word in ["july", "jul"]:
            return 7
        if word in ["august", "aug"]:
            return 8
        if word in ["september", "sept"]:
            return 9
        if word in ["october", "oct"]:
            return 10
        if word in ["november", "nov"]:
            return 11
        if word in ["december", "dec"]:
            return 12

        return None

    def to_decimal(self, x):
        return Decimal(x.replace(",", ""))

    def is_number(self, s, separator=None):
        if separator is not None:
            s = s.replace(separator, '')

        try:
            self.to_decimal(s)
            return True
        except:
            return False

    def separate_thousands(self, number, separator=''):
        number = f"{number:f}"
        if separator == '':
            return number

        num, _, frac = number.partition('.')
        num = re.sub(r'(\d{3})(?=\d)', r'\1' + separator, num[::-1])[::-1]
        if frac:
            num += '.' + frac
        return num
