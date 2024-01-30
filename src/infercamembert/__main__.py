import argparse
import json

from infercamembert.inference import infer
from infercamembert.labels import Labels
from infercamembert.parameters import DEFAULT_THRESHOLD, ModelParameters


def main(args):
    """
    Compute the inference of the input for the passed CamemBERT fine-tuned model
    """
    if not args.input or not args.dictionary or not args.model:
        raise Exception("missing mandatory parameters")

    with open(args.input, "r") as inputfile:
        inputs = json.load(inputfile)

    with open(args.dictionary, "r") as labelfile:
        labels = Labels(json.load(labelfile))

    params = ModelParameters(args.model, args.threshold)
    output = infer(inputs, labels, params)

    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    print(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="JSONL file to use as inputs")
    parser.add_argument("-d", "--dictionary", help="JSON file describing the labels")
    parser.add_argument("-m", "--model", help="name of the HuggingFace model")
    parser.add_argument(
        "-t",
        "--threshold",
        help=f"threshold value [default={DEFAULT_THRESHOLD}]",
        default=DEFAULT_THRESHOLD,
    )
    args = parser.parse_args()

    main(args)
