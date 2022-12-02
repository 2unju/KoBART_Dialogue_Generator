import torch.nn as nn
from transformers import BartForConditionalGeneration


class DialogueGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(config["model_path"]).to(config["device"])
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.sep_token = "<unused0>"

        self.tok = config["tok"]
        self.pad_token_id = self.tok.pad_token_id

    def forward(self, input_ids, decoder_input_ids, labels):
        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels, return_dict=True)
