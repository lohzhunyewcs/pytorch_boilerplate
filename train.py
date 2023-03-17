from eval import evaluate
import torch
from torch.utils.data import DataLoader
from task_type import TextTaskType, VisionTaskType
from typing import Union
import copy
import time

def print_and_log(str, log_file):
    raise NotImplementedError

def post_process(
    task: Union[TextTaskType, VisionTaskType],
    num_class: int,
    model_output: torch.Tensor
):
    if task == VisionTaskType.ImageClassification:
        if num_class == 1:
            cat_output = (torch.sigmoid(model_output) >= 0.5).int()
        else:
            cat_output = torch.argmax(torch.softmax(model_output, dim=1), dim=1)

    return cat_output

def train(
    model: torch.nn.Module,
    mode_to_dataloaders: dict[str, DataLoader],
    mode_to_datasize: dict[str, int],
    task: Union[TextTaskType, VisionTaskType],
    num_class: int,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.optimizer.Optimizer,
    device: torch.device,
    model_save_folder_path: str,
    scheduler: torch.optim.lr_scheduler=None,
    n_epochs: int=5,
):
    best_acc = 0
    model = model.to(device)

    overall_start_time = time.time()

    for n_epoch in range(n_epochs):
        epoch_start_time = time.time()
        print(f'-' * 10)
        print(f'On {n_epoch = }')
        for mode, dataloader in mode_to_dataloaders.items():
            if mode ==  "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            datasize = mode_to_datasize[mode]

            with torch.set_grad_enabled(mode == "train"):
                for inputs, gts in dataloader:
                    inputs = inputs.to(device)
                    gts = gts.to(device)
                    outputs = model(inputs)

                    predicted_cat = post_process(task, num_class, outputs)
                    
                    loss = criterion(outputs, gts)

                    if mode == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predicted_cat == gts)

            if mode == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects.double() / datasize

            print(f'{mode} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if mode == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        model_save_path = f'{model_save_folder_path}/{n_epoch}.pt'
        torch.save(model.state_dict(), model_save_path)

        print(f'Current {best_acc = }')

        print(f'Total time taken for this epoch = {time.time() - epoch_start_time}')
        print(f'Total time taken so far = {time.time() - overall_start_time}')