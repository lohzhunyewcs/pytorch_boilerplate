from eval import evaluate
import torch
from torch.utils.data import DataLoader
from task_type import TextTaskType, VisionTaskType
from typing import Union
import copy
import time
from metrics import BaseMetrics
import os

from tqdm import tqdm
import sys

def print_and_log(str, log_file):
    print(f'{str}')
    with open(log_file, 'a') as log:
        log.write(f'{str}\n')

def post_process(
    task: Union[TextTaskType, VisionTaskType],
    num_class: int,
    model_output: torch.Tensor
):
    if task == VisionTaskType.ImageClassification:
        if num_class == 1:
            cat_output = (torch.sigmoid(model_output) >= 0.5).int()
        else:
            cat_output = torch.argmax(model_output, dim=1)

    return cat_output

def train(
    model: torch.nn.Module,
    mode_to_dataloaders: dict[str, DataLoader],
    mode_to_datasize: dict[str, int],
    task: Union[TextTaskType, VisionTaskType],
    num_class: int,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_save_folder_path: str,
    metrics: list[BaseMetrics],
    scheduler: torch.optim.lr_scheduler=None,
    n_epochs: int=5,
):
    os.makedirs(model_save_folder_path, exist_ok=True)
    log_file_path = f'{model_save_folder_path}/log_file.txt'
    # Empty log file
    with open(log_file_path, 'w') as log_file:
        pass

    best_acc = 0
    best_epoch = 0

    # check if torch is version 2.0 and is not windows
    if torch.__version__ >= "2.0.0" and os.name != "nt":
        model = torch.compile(model)
    model = model.to(device)

    overall_start_time = time.time()

    for n_epoch in range(n_epochs):
        epoch_start_time = time.time()
        print_and_log(f'-' * 10, log_file_path)
        print_and_log(f'On {n_epoch = }', log_file_path)
        for mode, dataloader in mode_to_dataloaders.items():
            optimizer.zero_grad()
            if mode == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            datasize = mode_to_datasize[mode]

            with torch.set_grad_enabled(mode == "train"):
                with tqdm(
                    dataloader,
                    desc=mode,
                    file=sys.stdout,
                    disable=False,
                ) as iterator:
                    for inputs, gts in iterator:
                        inputs = inputs.to(device)
                        gts = gts.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        outputs = model(inputs)

                        predicted_cat = post_process(task, num_class, outputs)
                        
                        loss = criterion(outputs, gts)

                        if mode == "train":
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(predicted_cat == gts.data)

            # if mode == "train" and scheduler is not None:
            #     scheduler.step()

            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects.double() / datasize

            print_and_log(f'{mode} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', log_file_path)

            if mode == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = n_epoch

        model_save_path = f'{model_save_folder_path}/{n_epoch}.pt'
        torch.save(model.state_dict(), model_save_path)

        print_and_log(f'Current {best_acc = } from {best_epoch = }', log_file_path)

        print_and_log(f'Total time taken for this epoch = {time.time() - epoch_start_time}', log_file_path)
        print_and_log(f'Total time taken so far = {time.time() - overall_start_time}', log_file_path)