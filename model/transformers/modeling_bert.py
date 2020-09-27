# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import torch
import torch.utils.checkpoint

import transformers.modeling_bert as bert


# ugly hack against "None of the inputs have requires_grad=True. Gradients will be None"
def checkpoint(module, *args, **kwargs):
    dummy = torch.empty(1, requires_grad=True)
    return torch.utils.checkpoint.checkpoint(lambda d, *a, **k: module(*a, **k), dummy, *args, **kwargs)


class BertEncoder(bert.BertEncoder):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # ---- < changed lines > ----
            layer_outputs = checkpoint(
                layer_module, hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            # ---- </changed lines > ----

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertModel(bert.BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.init_weights()
