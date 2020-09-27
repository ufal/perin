from functools import reduce
from torchtext.data import Example


def example_from_json(obj, fields):
    ex = Example()
    for key, vals in fields.items():
        if vals is not None:
            if not isinstance(vals, list):
                vals = [vals]
            for val in vals:
                # for processing the key likes 'foo.bar'
                name, field = val
                ks = key.split(".")

                def reducer(obj, key):
                    if isinstance(obj, list):
                        results = []
                        for data in obj:
                            if key not in data:
                                # key error
                                raise ValueError("Specified key {} was not found in " "the input data".format(key))
                            else:
                                results.append(data[key])
                        return results
                    else:
                        # key error
                        if key not in obj:
                            raise ValueError("Specified key {} was not found in " "the input data".format(key))
                        else:
                            return obj[key]

                v = reduce(reducer, ks, obj)
                setattr(ex, name, field.preprocess(v))
    return ex
