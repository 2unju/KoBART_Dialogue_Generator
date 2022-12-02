import os
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DialogueGenerator
from dataloader import DGDataset
from arguments import get_inference_args
from setting import inference_setting


def inference(model_path, outpath, model_config, data_config):
    # model, tokenizer init
    model = DialogueGenerator(model_config)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model", model_path)))
    model.eval()
    tokenizer = model_config["tok"]

    outpath = os.path.join(os.path.dirname(__file__), outpath)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # dataset
    dev_dataset = DGDataset(data_config)
    dev_dataloader = DataLoader(dev_dataset, data_config["batch_size"])

    def del_underbar(tokens):
        return [re.sub("▁", "", t) for t in tokens]

    origin = open(os.path.join(outpath, "reference.txt"), "w", encoding="utf-8")
    with open(os.path.join(outpath, 'hypothesis.txt'), 'w', encoding='utf-8') as f:
        for step_index, batch_data in tqdm(enumerate(dev_dataloader), f"[INFERENCE]", total=len(dev_dataloader)):

            input_ids, decoder_input_ids, labels = tuple(value.to(model_config["device"]) for value in batch_data.values()
                                                         if type(value) != list)

            output = model.model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_length=100, num_beams=5)
            # output = model.model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_length=128,
            #                               num_beams=1, min_length=10, length_penalty=1.2, no_repeat_ngram_size=2)
            for o, res in zip(output, batch_data["origin_response"]):
                output = tokenizer.decode(o, skip_special_tokens=True)
                f.write(" ".join(tokenizer.tokenize(output)) + '\n')
                origin.write(" ".join(tokenizer.tokenize(res)) + "\n")
    origin.close()


if __name__ == "__main__":
    args = get_inference_args()
    model_config, data_config = inference_setting(args)
    inference(args.saved_model_path, args.outpath, model_config, data_config)
