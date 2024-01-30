# infer-camembert
_Python implementation for text classification inference with CamemBERT fine-tuned models_


![PyPI](https://img.shields.io/pypi/v/infer-camembert)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/cyrildever/infer-camembert)
![GitHub last commit](https://img.shields.io/github/last-commit/cyrildever/infer-camembert)
![GitHub issues](https://img.shields.io/github/issues/cyrildever/infer-camembert)
![GitHub](https://img.shields.io/github/license/cyrildever/infer-camembert)

This is a simple Python implementation for the inference step of a fine-tuned text classification model based on Transformer's `camembert-base` model and saved in HuggingFace&trade;.

### Usage

```console
$ pip install infer-camembert
```

For a private model, you must provide your HuggingFace token, either as an environment variable or under the `~/.huggingface` folder:
```console
$ HUGGINGFACE_TOKEN=<value> python3 -m infer-camembert --input=example.jsonl --dictionary=labels.json --model="your-public-or-private-model-on-huggingface" --threshold=0.1 > results.jsonl
```

Inputs must be in the form of a `dict` with the keys being your unique IDs and the values the text on which to perform inference, eg.
```json
{
  "id1": "Very nice time spent in a gorgeous site.",
  "id2": "Still a problem after three years: intolerable!!!!!!",
}
```
The same thing goes for the dictionary of labels where the keys should be your short custom labels and the value their corresponding long labels, eg.
```json
{
  "label0": "undefined",
  "label1": "pleasure",
  "label2": "fun",
  "label3": "anger",
}
```

The results are presented as an array of predictions per input line, eg.
```json
[
  {
    "id": "id1",
    "text": "Very nice time spent in a gorgeous site.", 
    "labels": [
      "pleasure",
      "fun"
    ]
  },
  {
    "id": "id2",
    "text": "Still a problem after three years: intolerable!!!!!!",
    "labels": [
      "anger"
    ]
  }
]
```

Used as a Python library:
```python
from infercamembert.inference import infer
from infercamembert.labels import Labels
from infercamembert.parameters import ModelParameters

inputs = {
    "id1": "Very nice time spent in a gorgeous site.",
    "id2": "Still a problem after three years: intolerable!!!!!!",
}
labels = Labels(
    {
        "label0": "undefined",
        "label1": "pleasure",
        "label2": "fun",
        "label3": "anger",
    }
)
params = ModelParameters("your-public-or-private-model-on-huggingface", 0.1)
outputs = infer(inputs, labels, params)
```


### License

This module is distributed under a MIT license. \
See the [LICENSE](./LICENSE) file.


<hr />
&copy; 2024 Cyril Dever. All rights reserved.