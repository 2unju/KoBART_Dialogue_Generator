import argparse


def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", default="gogamza/kobart-base-v1", type=str)
    parser.add_argument("--train-path", default=None, type=str)
    parser.add_argument("--valid-path", default=None, type=str)
    parser.add_argument("--outpath", default="./model", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max-len", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--scheduler", default="y", choices=["y", "n"], type=str)

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--ignore_index", default=-100, type=int)
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--warmup-ratio", default=0.1, type=float)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--train-log-interval", default=100, type=int)
    parser.add_argument("--validation-interval", default=100, type=int)

    return parser.parse_args()


def get_inference_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", default="gogamza/kobart-base-v1", type=str)
    parser.add_argument("--test-path", default=None, type=str)
    parser.add_argument("--outpath", default="inference", type=str)
    parser.add_argument("--saved-model-path", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max-len", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--ignore_index", default=-100, type=int)

    return parser.parse_args()

