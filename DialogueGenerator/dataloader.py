import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class DGDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        print(f"[DATALOADER] Load data from {config['file_path']}")
        self.tokenizer = config["tok"]
        self.max_len = config["max_len"]
        self.docs = pd.read_csv(config["file_path"], sep='\t', encoding='utf-8', names=["query", "response"])
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = config["ignore_idx"]

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['query'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['response'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_),
                'origin_response': instance["response"]}

    def __len__(self):
        return self.len
