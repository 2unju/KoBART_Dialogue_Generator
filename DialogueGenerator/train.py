import os
import math
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from model import DialogueGenerator
from dataloader import DGDataset
from arguments import get_train_args
from utils import seed_everything
from setting import train_setting

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train(
        model: DialogueGenerator,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        device: torch.device,
        optimizer,
        scheduler,
        epochs,
        grad_clip,
        train_log_interval,
        validation_interval
):
    min_loss = None

    model.train()
    loss_list_between_log_interval = []
    for epoch_id in range(epochs):
        for step_index, batch_data in tqdm(enumerate(train_dataloader), f"[TRAIN] EP:{epoch_id}",
                                           total=len(train_dataloader)):
            global_step = len(train_dataloader) * epoch_id + step_index + 1
            optimizer.zero_grad()

            input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values()
                                                         if type(value) != list)

            model_outputs = model.forward(input_ids, decoder_input_ids, labels)

            model_outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()

            # for logging
            loss_list_between_log_interval.append(model_outputs.loss.item())

            if global_step % train_log_interval == 0:
                mean_loss = np.mean(loss_list_between_log_interval)
                logger.info(
                    f"EP:{epoch_id} global_step:{global_step} "
                    f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
                )
                loss_list_between_log_interval.clear()

            if global_step % validation_interval == 0:
                dev_loss = _validate(model, dev_dataloader, device, logger, global_step)
                if not min_loss or min_loss > dev_loss:
                    if not os.path.exists(os.path.join(os.path.dirname(__file__), "model")):
                        os.mkdir(os.path.join(os.path.dirname(__file__), "model"))
                    min_loss = dev_loss
                    state_dict = model.state_dict()
                    model_path = os.path.join(os.path.dirname(__file__), 'model', f"loss.{dev_loss:.4f}.pth")
                    logger.info(f"[TRAIN] global_step: {global_step} model saved at {model_path}")
                    torch.save(state_dict, model_path)


def _validate(
        model: DialogueGenerator,
        dev_dataloader: DataLoader,
        device: torch.device,
        logger: logging.Logger,
        global_step: int,
):
    model.eval()
    loss_list = []
    for batch_data in tqdm(dev_dataloader, desc="[EVAL]"):
        with torch.no_grad():
            input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values()
                                                         if type(value) != list)
            model_outputs = model.forward(input_ids, decoder_input_ids, labels)
            loss_list.append(model_outputs.loss.item())

    mean_loss = np.mean(loss_list)
    logger.info(f"[EVAL] global_step:{global_step} loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}")
    model.train()

    return mean_loss


def run(args):
    model_config, data_config = train_setting(args)
    model = DialogueGenerator(model_config)

    train_config = data_config
    train_config["file_path"] = data_config["train_path"]
    trainset = DGDataset(train_config)
    trainloader = DataLoader(trainset, args.batch_size, shuffle=True)

    valid_config = data_config
    valid_config["file_path"] = data_config["valid_path"]
    validset = DGDataset(valid_config)
    validloader = DataLoader(validset, args.batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)

    data_len = len(trainloader)
    num_train_steps = int(data_len / args.batch_size * args.epoch)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    if args.scheduler == "y":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)
    else:
        scheduler = None

    logging.info(f'data length {data_len}')
    logging.info(f'num_train_steps : {num_train_steps}')
    logging.info(f'num_warmup_steps : {num_warmup_steps}')

    device = args.device
    train(model=model,
          train_dataloader=trainloader,
          dev_dataloader=validloader,
          device=device,
          optimizer=optimizer,
          scheduler=scheduler,
          epochs=args.epoch,
          grad_clip=args.grad_clip,
          train_log_interval=args.train_log_interval,
          validation_interval=args.validation_interval)


if __name__ == "__main__":
    args = get_train_args()
    seed_everything(args.seed)
    run(args)
