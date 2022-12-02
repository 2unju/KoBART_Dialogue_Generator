import os
from transformers import PreTrainedTokenizerFast


def train_setting(args):
    if not args.train_path:
        args.train_path = os.path.join(os.path.dirname(__file__), "data", f"train.tsv")
    if not args.valid_path:
            args.valid_path = os.path.join(os.path.dirname(__file__), "data", f"valid.tsv")
    tok = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    tok.add_special_tokens({"sep_token": "<unused0>"})

    model_config = dict()
    model_config["model_path"] = args.model_path
    model_config["device"] = args.device
    model_config["tok"] = tok

    data_config = dict()
    data_config["tok"] = tok
    data_config["max_len"] = args.max_len
    data_config["ignore_idx"] = args.ignore_index
    data_config["train_path"] = args.train_path
    data_config["valid_path"] = args.valid_path
    return model_config, data_config


def inference_setting(args):
    if not args.test_path:
        args.test_path = os.path.join(os.path.dirname(__file__), "data", "test.tsv")

    tok = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    tok.add_special_tokens({"sep_token": "<unused0>"})

    model_config = dict()
    model_config["model_path"] = args.model_path
    model_config["device"] = args.device
    model_config["tok"] = tok

    data_config = dict()
    data_config["tok"] = tok
    data_config["max_len"] = args.max_len
    data_config["ignore_idx"] = args.ignore_index
    data_config["file_path"] = args.test_path
    data_config["batch_size"] = args.batch_size

    return model_config, data_config
