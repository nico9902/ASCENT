import argparse
from pathlib import Path
import os
from openpyxl import load_workbook
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random


def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()


def seed_all(seed):
    if not seed:
        seed = 0
    print("Using Seed : ", seed)

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Empty and create direcotory
def create_dir(dir):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)


def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


def save_results(sheet_name, file, table, index=False, header=True):
    try:
        book = load_workbook(file)
        writer = pd.ExcelWriter(file)
        writer.book = book
    except FileNotFoundError:
        writer = pd.ExcelWriter(file)
    table.to_excel(writer, sheet_name=sheet_name, index=index, header=header)
    writer.save()
    writer.close()

def check_na(var):
    if var == "None":
        return None
    else:
        return var
    
def check_active_fn(active_fn):
    if active_fn == "relu":
        return nn.ReLU()
    elif active_fn == "sigmoid":
        return nn.Sigmoid()
    elif active_fn == "tanh":
        return nn.Tanh()
    elif active_fn == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)  # You can adjust the negative slope as needed
    elif active_fn == "elu":
        return nn.ELU()
    elif active_fn == "softmax":
        return nn.Softmax(dim=-1)  # Specify dimension if needed
    else:
        raise ValueError(f"Activation function '{active_fn}' not supported.")
