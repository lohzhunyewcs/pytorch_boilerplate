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
    model_output: torch.Tensor,
    gts: torch.Tensor,
    threshold: float=0.5
):
    if task == VisionTaskType.ImageClassification:
        if num_class == 1:
            cat_output = (torch.sigmoid(model_output) >= threshold).int()
        else:
            cat_output = torch.argmax(model_output, dim=1)
    else:
        raise NotImplementedError

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

    # # check if torch is version 2.0 and is not windows
    # if torch.__version__ >= "2.0.0" and os.name != "nt":
    #     model = torch.compile(model)
    model = model.to(device)

    overall_start_time = time.time()

    for n_epoch in range(n_epochs):
        epoch_start_time = time.time()
        print_and_log(f'-' * 10, log_file_path)
        print_and_log(f'On {n_epoch = }', log_file_path)
        for mode, dataloader in mode_to_dataloaders.items():
            # for metric in metrics:
            #     metric.reset()

            if mode == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            datasize = mode_to_datasize[mode]

            with tqdm(
                dataloader,
                desc=mode,
                file=sys.stdout,
                disable=False,
            ) as iterator:
                iterated_data_size = 0
                for inputs, gts in iterator:
                    inputs = inputs.to(device)
                    gts = gts.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                        
                    with torch.set_grad_enabled(mode == "train"):
                        outputs = model(inputs)

                        predicted_cat = post_process(task, num_class, outputs, gts)
                        
                        loss = criterion(outputs, gts)

                        for metric in metrics:
                            metric.add(predicted_cat, gts)

                        if mode == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predicted_cat == gts.data)

                    iterated_data_size += inputs.size(0)

                    it_desc = f"{mode} Loss = {running_loss / iterated_data_size:.4f}, Acc = {running_corrects.double() / iterated_data_size:.4f}"
                    
                    for metric in metrics:
                        it_desc += f' {metric.__name__}: {metric.compute():.4f}'
                    iterator.set_description(it_desc)

            if mode == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects.double() / datasize

            print_string = f'{mode} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
            for metric in metrics:
                print_string += f' {metric.__name__}: {metric.compute():.4f}'
                metric.reset()
            # print_and_log(f'{mode} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', log_file_path)
            print_and_log(print_string, log_file_path)
            

            if 'val' in mode and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = n_epoch

        model_save_path = f'{model_save_folder_path}/{n_epoch}.pt'
        torch.save(model.state_dict(), model_save_path)

        print_and_log(f'{mode = } current {best_acc = } from {best_epoch = }', log_file_path)

        print_and_log(f'Total time taken for this epoch = {time.time() - epoch_start_time}', log_file_path)
        print_and_log(f'Total time taken so far = {time.time() - overall_start_time}', log_file_path)


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
