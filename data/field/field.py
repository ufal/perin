import torch
import torchtext
from collections import Counter, OrderedDict


# small change of vocab building to correspond to our version of Dataset
class Field(torchtext.data.Field):
    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, torch.utils.data.Dataset):
                sources += [arg.get_examples(name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)

        specials = list(
            OrderedDict.fromkeys(
                tok
                for tok in [self.unk_token, self.pad_token, self.init_token, self.eos_token] + kwargs.pop("specials", [])
                if tok is not None
            )
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def process(self, example, device=None):
        if self.include_lengths:
            example = example, len(example)
        tensor = self.numericalize(example, device=device)
        return tensor

    def numericalize(self, ex, device=None):
        if self.include_lengths and not isinstance(ex, tuple):
            raise ValueError("Field has include_lengths set to True, but input data is not a tuple of (data batch, batch lengths).")

        if isinstance(ex, tuple):
            ex, lengths = ex
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                ex = [self.vocab.stoi[x] for x in ex]
            else:
                ex = self.vocab.stoi[ex]

            if self.postprocessing is not None:
                ex = self.postprocessing(ex, self.vocab)
        else:
            numericalization_func = self.dtypes[self.dtype]

            if not self.sequential:
                ex = numericalization_func(ex) if isinstance(ex, str) else ex
            if self.postprocessing is not None:
                ex = self.postprocessing(ex, None)

        var = torch.tensor(ex, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var
