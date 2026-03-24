# import libraries
import numpy as np
import wandb
import time
import copy
import pandas as pd
import os
import gc
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score

import torch
import torch.nn as nn
from torchvision import models

import src.utils.util_general as util_general
import src.model.networks as networks
from src.utils.util_eval import ct_index

from MedViT.MedViT import MedViT_small, MedViT_base, MedViT_large

def set_parameter_requires_grad(model, freeze=False, half_freeze=False, unfreeze_last=False):
    """
    Adjusts the `requires_grad` attribute of model parameters based on the specified mode.

    Args:
        model (torch.nn.Module): The model whose parameters need adjustment.
        freeze (bool): If True, freezes all parameters in the model.
        half_freeze (bool): If True, freezes the first half of the model's layers.
        unfreeze_last (bool): If True, freezes all layers except the last one.

    Raises:
        ValueError: If more than one mode (freeze, half_freeze, unfreeze_last) is set to True.
    """
    # Ensure only one mode is active
    active_modes = [freeze, half_freeze, unfreeze_last]
    if sum(active_modes) > 1:
        raise ValueError("Only one of 'freeze', 'half_freeze', or 'unfreeze_last' can be True.")

    if freeze:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

    elif half_freeze:
        # Freeze the first half of the layers
        layers = list(model.modules())
        num_layers_to_freeze = len(layers) // 2
        for layer in layers[:num_layers_to_freeze]:
            for param in layer.parameters(recurse=False):
                param.requires_grad = False

    elif unfreeze_last:
        # Freeze all layers except the last one
        layers = list(model.modules())
        for layer in layers[:-1]:
            for param in layer.parameters(recurse=False):
                param.requires_grad = False
    
def model_rgb2gray(model): 
   """
       Function to convert the first layer of a model to process grayscale images
   """
   # identify first layer 
   first_layer = model 
   while len(list(first_layer.children())) > 1: 
      first_layer = list(first_layer.children())[0] 

   # convert first layer to process grayscale image 
   first_layer.in_channels = 1        
   first_layer.weight = torch.nn.Parameter(first_layer.weight.sum(1, keepdim=True))


def initialize_model(model_name, backbone_output_size, cfg_model):
    """
        Initialize the model
    """
    # check if pretrained model is used
    if cfg_model["pretrained"] == "None":
        pretrained = None
    else:
        pretrained = cfg_model["pretrained"]

    # initialize model
    if model_name == "alexnet":
        model = models.alexnet(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg11":
        model = models.vgg11(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(weights=pretrained)
        # replace classification block with one fully connection layer
        # model.classifier = nn.Linear(25088, backbone_output_size)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg13":
        model = models.vgg13(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg13_bn":
        model = models.vgg13_bn(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg16":
        model = models.vgg16(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg19":
        model = models.vgg19(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(weights=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "resnet18":
        model = models.resnet18(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "resnet34":
        model = models.resnet34(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "resnet50":
        model = models.resnet50(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "resnet101":
        model = models.resnet101(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "resnet152":
        model = models.resnet152(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=pretrained)
        model.classifier[0] = torch.nn.Dropout(p=cfg_model["dropout"], inplace=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b5":
        model = models.efficientnet_b5(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b6":
        model = models.efficientnet_b6(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(weights=pretrained)
        model.classifier._modules["1"] = nn.Conv2d(512, backbone_output_size, kernel_size=(1, 1))
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "swin_t":
        model = models.swin_t(weights=pretrained)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "vit":
        model = models.vit_b_16(weights=pretrained)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=pretrained)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    elif model_name == "MedViT_small":
        model = MedViT_small(num_classes = 1000)
        if pretrained:
            checkpoint = torch.load("/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/deep-lung/MedViT/MedViT_small_im1k.pth")  #("/Users/domenicopaolo/Documents/PhD AI/Projects/deep-lung/MedViT/MedViT_small_im1k.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        model.proj_head = nn.Linear(in_features=1024, out_features=backbone_output_size)
        set_parameter_requires_grad(model, cfg_model["freeze"], cfg_model["half_freeze"], cfg_model["unfreeze_last"])
    else:
        print("Invalid model name, exiting...")
        exit()

    # sum pretrained RGB convolutional filters
    model_rgb2gray(model=model)

    return model


def train_model(model, criterion, optimizer, scheduler, warmup_scheduler, warmup_period, exp_name, data_loaders, model_dir, device, cfg_trainer, run_wandb=True):
    """
        Function to train the model
    """
    # initialize training parameters
    epochs = cfg_trainer["max_epochs"]
    save_best_loss = cfg_trainer['best_loss']
    early_stoppping = cfg_trainer["early_stopping"]

    # initialize best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    if cfg_trainer['best_loss']:
        best_loss = np.Inf
    else:
        best_metric = -np.Inf

    # define history for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_cindex': [], 'val_cindex': []}
    
    # training loop
    epochs_no_improve = 0
    early_stop = False
    since = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1), flush=True)
        print('-' * 10, flush=True)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            # initialize predictions
            running_loss = 0.0
            surv_times = []
            predictions = []
            targets = []
            # iterate over data
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for inputs, times, labels, mask1, mask2, _ in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    times = times.to(device)
                    mask1 = mask1.to(device)
                    mask2 = mask2.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward (track history if only in train)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())
                        times = torch.unsqueeze(times, -1)
                        labels = torch.unsqueeze(labels, -1)
                        loss = criterion(outputs, times, labels, mask1, mask2)

                        # detach and move to CPU for numpy operations
                        preds_np = outputs.detach().cpu().numpy()
                        labels_np = labels.detach().cpu().numpy()
                        times_np = times.detach().cpu().numpy()
                            
                        for output, label, surv_time in zip(preds_np, labels_np, times_np):
                            surv_times.append(surv_time)
                            targets.append(label)
                            predictions.append(output)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    pbar.update(inputs.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            # concatenate predictions over batches
            predictions = np.concatenate(predictions, axis=0)
            predictions = np.expand_dims(predictions, axis=1)
            surv_times = np.concatenate(surv_times, axis=0)
            targets = np.concatenate(targets, axis=0)

            # compute metrics
            epoch_cindex = ct_index(predictions, surv_times, targets, 0)

            # if run_wandb:
            #     if phase == 'train':
            #         wandb.log({'train_loss': epoch_loss, 'train_cindex': epoch_cindex}, step=epoch)
            #     else:
            #         wandb.log({'val_loss': epoch_loss, 'val_cindex': epoch_cindex}, step=epoch)

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    if phase == 'val':
                        scheduler.step()

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_cindex'].append(epoch_cindex)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_cindex'].append(epoch_cindex)

            print('{} Loss: {:.4f} ctindex: {:.4f}'.format(phase, epoch_loss, epoch_cindex), flush=True)

            # if phase == 'val':
            #     print(evaluate(model, data_loaders['test'], device, classification=False))

            # deep copy the model
            epoch_metric = epoch_cindex
            if phase == 'val':
                if save_best_loss:
                    if epoch_loss < best_loss:
                        best_epoch = epoch
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # trigger early stopping
                        if epochs_no_improve >= early_stoppping:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break
                else:
                    if epoch_metric > best_metric:
                        best_epoch = epoch
                        best_metric = epoch_metric
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # trigger early stopping
                        if epochs_no_improve >= early_stoppping:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    if save_best_loss:
        print('Best val loss: {:4f}'.format(best_loss))
    else:
        print('Best ct-index: {:4f}'.format(best_metric))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model, os.path.join(model_dir, "%s.pt" % exp_name))

    # plot training curves (saved under model_dir/model_name)
    try:
        plot_training(history, exp_name, model_dir, criterion)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    # explicitly delete unused variables and clear cache
    del best_model_wts
    gc.collect()
    torch.cuda.empty_cache()

    return model


def plot_training(history, model_name, plot_training_dir, criterion):
    """
        Function to plot the training results
    """
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
   
    # training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(type(criterion).__name__)
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"))
    plt.show()
    
    # training result cindex
    plt.figure(figsize=(8, 6))
    for c in ['train_cindex', 'val_cindex']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cindex')
    plt.title('Training and Validation cindex')
    plt.savefig(os.path.join(model_plot_dir, "cindex"))
    plt.show()

def evaluate(model, data_loader, device):
    """
        Function to evaluate the model
    """
    # initialize predictions
    predictions = []
    targets = []
    surv_times = []

    # test loop
    model.eval()
    with torch.no_grad():
        for inputs, times, labels, mask1, mask2, pids in tqdm(data_loader):
            inputs, times, labels, mask1, mask2 = inputs.to(device), times.to(device), labels.to(device), mask1.to(device), mask2.to(device)

            # prediction
            outputs = model(inputs.float())
            print(outputs)

            # detach and move to CPU for numpy operations
            preds_np = outputs.detach().cpu().numpy()
            labels_np = torch.unsqueeze(labels, -1).detach().cpu().numpy()
            times_np = torch.unsqueeze(times, -1).detach().cpu().numpy()

            for output, label, surv_time in zip(preds_np, labels_np, times_np):
                surv_times.append(surv_time)
                targets.append(label)
                predictions.append(output)

    # concatenate predictions over batches
    predictions = np.array(predictions)
    targets = np.array(targets)
    surv_times = np.array(surv_times)
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.expand_dims(predictions, axis=1)
    surv_times = np.concatenate(surv_times, axis=0)
    targets = np.concatenate(targets, axis=0)
    test_cindex = ct_index(predictions, surv_times, targets, 0)
    test_results = {"cindex": test_cindex}

    return test_results


def predict(model, data_loader, device):
    """
        Function to predict the model
    """
    # initialize predictions
    surv_times = []
    predictions = []
    targets = []

    # predict loop
    model.eval()
    with torch.no_grad():
        for inputs, times, labels, mask1, mask2, _ in tqdm(data_loader):
            inputs, times, labels, mask1, mask2 = inputs.to(device), times.to(device), labels.to(device), mask1.to(device), mask2.to(device)

            # prediction
            outputs = model(inputs.float())

            # detach and move to CPU for numpy operations
            preds_np = outputs.detach().cpu().numpy()
            labels_np = torch.unsqueeze(labels, -1).detach().cpu().numpy()
            times_np = torch.unsqueeze(times, -1).detach().cpu().numpy()

            for output, label, surv_time in zip(preds_np, labels_np, times_np):
                surv_times.append(surv_time)
                targets.append(label)
                predictions.append(output)

    return predictions, targets, surv_times


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def get_predictions(prediction_dir, fold_list, steps):
    results = pd.DataFrame()

    for fold in fold_list:

        fold_path = os.path.join(prediction_dir, str(fold))

        for Fset in steps:

            preds_path = os.path.join(fold_path, f"prediction_{Fset}.xlsx")
            probs_path = os.path.join(fold_path, f"probability_{Fset}.xlsx")

            preds = pd.read_excel(preds_path, engine="openpyxl", index_col=0)
            probs = pd.read_excel(probs_path, engine="openpyxl", index_col=0)

            cols_to_drop = [col for col in probs.columns.to_list() if col.endswith("_0")]
            probs = probs.drop(cols_to_drop + ["True"], axis=1)

            clear_names = [col.replace("_1", "") for col in probs.columns.to_list()]
            new_names = pd.MultiIndex.from_product([clear_names, ["probability"]])

            probs.columns = new_names

            new_names = pd.MultiIndex.from_product([preds.columns.to_list(), ["prediction"]])

            preds.columns = new_names

            preds = pd.concat([preds, probs], axis=1)

            for classifier in preds.columns.levels[0]:
                preds[(classifier, "label")] = preds[("True", "prediction")]

            preds = preds.drop([("True", "label"), ("True", "prediction")], axis=1)

            preds = preds.assign(fold=fold, Fset=Fset)

            preds = preds.reset_index().set_index(["Fset", "fold", "ID"])

            results = pd.concat([results, preds], axis=0)

    return results.sort_index(axis=1)

def compute_performance(preds, targs, times):
    # convert to numpy arrays
    preds = np.array(preds)
    targs = np.array(targs)
    times = np.array(times)

    # concatenate predictions over batches
    preds = np.concatenate(preds, axis=0)
    preds = np.expand_dims(preds, axis=1)
    targs = np.concatenate(targs, axis=0)
    times = np.concatenate(times, axis=0)

    # compute metrics
    test_cindex = ct_index(preds, times, targs, 0)
    test_results = {"cindex": test_cindex}
    
    return test_results