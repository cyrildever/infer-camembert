import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from infercamembert.labels import Labels
from infercamembert.parameters import ModelParameters
from infercamembert._process import get_preds_from_logits


def infer(inputs: dict, labels: Labels, params: ModelParameters) -> list[dict]:
    """
    Make predictions for the passed inputs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(ModelParameters.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        params.model_name, token=hf_token
    ).to(device)

    # Data prep
    input_texts = list(inputs.values())
    encoded = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    ).to(device)

    # Make the predictions
    logits = model(**encoded).logits.cpu().detach().numpy()

    # Decode the results
    preds = get_preds_from_logits(logits, params.threshold)
    id2key = labels.get_id2key()
    decoded_preds = [[id2key[i] for i, l in enumerate(row) if l == 1] for row in preds]

    ids = list(inputs.keys())
    outputs = list()
    for idx, (text, pred) in enumerate(zip(input_texts, decoded_preds)):
        result = {
            "id": ids[idx],
            "text": text,
            "labels": [labels.get_label_from_dictionary(l) for l in pred],
        }
        outputs.append(result)

    return outputs
