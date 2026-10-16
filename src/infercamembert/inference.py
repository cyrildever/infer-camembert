import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from infercamembert import Labels, ModelParameters
from infercamembert._process import get_preds_from_logits


DEFAULT_BATCH_SIZE = 64
MAX_LENGTH = 256


class LoadedModel:
    """
    A tokenizer and a fine-tuned model ready for inference, loaded once and reused across calls to `infer`
    """

    def __init__(self, params: ModelParameters, device: str | None = None):
        self.model_name = params.model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(ModelParameters.BASE_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                params.model_name, token=hf_token
            )
            .to(self.device)
            .eval()
        )

    def __str__(self) -> str:
        return f"LoadedModel(model_name='{self.model_name}', device='{self.device}')"

    def logits(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Returns the raw logits of the passed texts, one row per text
        """
        if len(texts) == 0:
            return np.empty((0, self.model.config.num_labels))

        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), max(1, batch_size)):
                chunk = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                rows.append(self.model(**encoded).logits.cpu().numpy())
        return np.concatenate(rows, axis=0)


def load_model(params: ModelParameters, device: str | None = None) -> LoadedModel:
    """
    Load the tokenizer and the model of the passed parameters once, to pass to `infer` as `loaded`
    """
    return LoadedModel(params, device)


def infer(
    inputs: dict,
    labels: Labels,
    params: ModelParameters,
    loaded: LoadedModel | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """
    Make predictions for the passed inputs

    Without `loaded`, the model is loaded for this call only: pass the result of `load_model` to reuse it.
    """
    if loaded is None or loaded.model_name != params.model_name:
        loaded = load_model(params)

    ids = list(inputs.keys())
    input_texts = list(inputs.values())

    logits = loaded.logits(input_texts, batch_size)

    # Decode the results
    preds = get_preds_from_logits(logits, params.threshold)
    id2key = labels.get_id2key()
    decoded_preds = [[id2key[i] for i, l in enumerate(row) if l == 1] for row in preds]

    outputs = list()
    for idx, (text, pred) in enumerate(zip(input_texts, decoded_preds)):
        result = {
            "id": ids[idx],
            "text": text,
            "labels": [labels.get_label_from_dictionary(l) for l in pred],
        }
        outputs.append(result)

    return outputs
