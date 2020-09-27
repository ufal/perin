#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import xml.etree.ElementTree as ET


VALLEX_PATHS = {
    "eng": "vallex_en.xml",
    "ces": "vallex_cz.xml",
}

FRAME_PREFIXES = {
    "eng": "en-v#",
    "ces": "v#",
}


class FramePredictor:
    def __init__(self, language):
        self.load(VALLEX_PATHS[language])
        self.frame_prefix = FRAME_PREFIXES[language]

    def load(self, path):
        self.forms = {}
        root = ET.parse(path).getroot()

        for form in root[1]:
            lemma = form.attrib["lemma"].lower()
            form_id = form.attrib["id"]
            self.forms[lemma] = {"id": form_id, "frames": []}
            for frame in form[0]:
                self.forms[lemma]["frames"].append(frame.attrib["id"][len(form_id):])
            assert len(self.forms[lemma]) > 0

    def predict(self, form, probabilities):
        if form not in self.forms:
            return "<NONE>"

        best_frame = None
        for frame in self.forms[form]["frames"]:
            if frame in probabilities and (best_frame is None or probabilities[frame] > best_frame[0]):
                best_frame = (probabilities[frame], frame)
            if frame not in probabilities and best_frame is None:
                best_frame = (0.0, frame)

        if best_frame is None:
            return "<NONE>"

        return self.frame_prefix + self.forms[form]["id"] + best_frame[1]
