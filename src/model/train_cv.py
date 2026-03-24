# import libraries
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import pandas as pd
import numpy as np
import collections
import yaml
# import wandb
import pytorch_warmup as warmup

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model
import src.model.networks as networks
import src.data.datasets as dataset
import src.model.collate as collate
from src.model.losses import DeepHitLoss

# empty the cache
torch.cuda.empty_cache()

# configuration file
args = util_general.get_args()
seed = args.seed
exp_name = args.exp_name
with open(args.cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# seed everything
util_general.seed_all(seed)

# parameters
backbone_output_size = cfg['backbone']['backbone_output_size']
mode = cfg['model']['mode']
cv = cfg['data']['cv']
fold_start = cfg['data']['fold_start']
pretrained = cfg['model']['pretrained']
augmentation = cfg['data']['augmentation']
fold_list = list(range(cv))
run_wandb = False #cfg['run_wandb']

# deephit parameters
input_dims = cfg['deephit']['input_dims']
network_settings = cfg['deephit']['network_settings']
num_Event = util_general.check_na(input_dims['num_Event'])
num_Category = util_general.check_na(input_dims['num_Category'])
network_settings['active_fn'] = util_general.check_active_fn(network_settings['active_fn'])
alpha = cfg['deephit']['alpha']
beta = cfg['deephit']['beta']

# device
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']

# create files and directories
model_dir = os.path.join(cfg['data']['model_dir'], exp_name)
util_general.create_dir(model_dir)
report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)
report_file = os.path.join(report_dir, 'report.xlsx')
plot_training_dir = os.path.join(report_dir, "training")
util_general.create_dir(plot_training_dir)

# main loop
if __name__=="__main__":
    print(f"Starting. CUDA={torch.cuda.is_available()} device={device}", flush=True)
    # loop over the folds
    results = collections.defaultdict(lambda: [])
    acc_cols = []
    acc_class_cols = collections.defaultdict(lambda: [])
    predictions = []
    targets = []
    surv_times = []
    for fold in range(fold_start, cv):
        print(f"Fold {fold+1}/{cv}", flush=True)
        # create datasets
        fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.csv' % step), delimiter=",") for step in ['train', 'val', 'test']}
        datasets = {step: dataset.ImgDataset(patientIDs=fold_data[step]['PatientID'].tolist(), cfg_data=cfg['data'], time=fold_data[step]['Time'].tolist(), label=fold_data[step]['Label'].tolist(), augmentation=augmentation, step=step, num_Event=num_Event, num_Category=num_Category) for step in ['train', 'val', 'test']}
        print(f"Dataset sizes: train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}", flush=True)

        # define data transform as a composition of Convert and Normalize
        # here, Normalize implements standardization using the computed mu and std of the training set
        # add Augmentation in training step if required
        mu = datasets['train'].mu
        std = datasets['train'].std
        print(f"Computed mu={mu} std={std}", flush=True)
        DataAugmentation = transforms.RandomApply([transforms.RandomRotation(20, fill=(0,)), transforms.RandomHorizontalFlip()], p=0.2)
        if cfg['data']['augmentation']:
            transform_train = transforms.Compose([DataAugmentation,
                                        collate.Convert(),
                                        transforms.Normalize(mean=mu, std=std)])
        else:
            transform_train = transforms.Compose([
                                        collate.Convert(),
                                        transforms.Normalize(mean=mu, std=std)])
        transform_test = transforms.Compose([collate.Convert(), transforms.Normalize(mean=mu, std=std)])
        datasets['train'].transform = transform_train
        for step in ['val', 'test']:
            datasets[step].transform = transform_test

        # create data loaders
        print("Creating DataLoaders...", flush=True)
        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['model']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker, collate_fn=collate.MyCollator(stage='train', mask_percentage=cfg['model']['mask_perc']), drop_last=True),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['model']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker, collate_fn=collate.MyCollator(stage='val', mask_percentage=cfg['model']['mask_perc']), drop_last=True),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['model']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker, collate_fn=collate.MyCollator(stage='test', mask_percentage=cfg['model']['mask_perc']), drop_last=True)}
        print("DataLoaders ready.", flush=True)

        # create directories
        model_fold_dir = os.path.join(model_dir, str(fold))
        util_general.create_dir(os.path.join(model_fold_dir, exp_name))
        plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
        util_general.create_dir(os.path.join(plot_training_fold_dir, exp_name))
        
        # select backbone for 2D networks
        if mode not in ["ResNet3D", "densenet121", "densenet161", "densenet169", "densenet201", "DINOv2"]:
            # define the backbone
            backbone_name = cfg['backbone']['backbone_name']

            # initialize backbone
            backbone = util_model.initialize_model(model_name=backbone_name, backbone_output_size=backbone_output_size, cfg_model=cfg['backbone'])
            
            # print backbone name
            print("%s%s%s" % ("*"*50, backbone_name, "*"*50), flush=True)
        else:
            backbone_name = None

        # initialize model
        # 2D networks
        if mode == "soft":
            model = networks.SoftAttentionModel(backbone=backbone, backbone_output_size=backbone_output_size, mlp_layer=cfg['model']['mlp'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings)
            # load pretrained weights (if needed)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{backbone_name}.pt"
                pretrained_dict = torch.load(pretrained_model_weights,weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        elif mode == "mean":
            model = networks.SoftAttentionModel(backbone=backbone, backbone_output_size=backbone_output_size, mlp_layer=cfg['model']['mlp'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings, fusion_criterion="mean")
        elif mode == "kron":
            model = networks.SoftAttentionModel(backbone=backbone, backbone_output_size=backbone_output_size, mlp_layer=cfg['model']['mlp'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings, fusion_criterion="kronecker")
        elif mode== "trans_encoder":
            model = networks.TransEncoderModel(backbone=backbone, embed_size=backbone_output_size, num_layers=cfg['model']['num_layers'], num_heads=cfg['model']['num_heads'], mlp_layer=cfg['model']['mlp'], conv_layer=cfg['model']['conv'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings)
        elif mode == "majority_voting":
            model = networks.VotingModel(backbone=backbone, backbone_output_size=backbone_output_size, num_classes=cfg['data']['num_classes'])
        # 3D networks
        elif mode == "ResNet3D":
            model = networks.ResNet183D(cfg, pretrained=True, input_dims=input_dims, network_settings=network_settings, embedding_size=backbone_output_size)
        elif mode in ["densenet121", "densenet161", "densenet169", "densenet201"]:
            model = networks.DenseNet3D(cfg, backbone_name=mode, input_dims=input_dims, network_settings=network_settings, embedding_size=backbone_output_size)
        elif mode == "DINOv2":
            model = networks.DINOv2(cfg, input_dims=input_dims, network_settings=network_settings, embedding_size=backbone_output_size)
        else:
            raise ValueError("Il valore di mode non è valido.")
        
        # move model to device
        model = model.to(device)
        print("number of gpus: ", torch.cuda.device_count())
        print("Model created and moved to device.", flush=True)
        if torch.cuda.device_count() > 1:
            print("number of gpus: ", torch.cuda.device_count())
            model = nn.DataParallel(model, [0,1])

        # loss function
        criterion = DeepHitLoss(alpha=alpha, beta=beta, num_Event=input_dims['num_Event'], num_Category=input_dims['num_Category']).to(device)
        
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=0.01)
        # lr_scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # warmup
        warmup_period = 10
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

        # train model
        print("Training the model...", flush=True)
        model = util_model.train_model(model=model, criterion=criterion, optimizer=optimizer,
                                                scheduler=scheduler, warmup_scheduler=warmup_scheduler, warmup_period=warmup_period, 
                                                exp_name=exp_name, data_loaders=data_loaders,
                                                model_dir=model_fold_dir, device=device,
                                                cfg_trainer=cfg['trainer'], run_wandb=run_wandb)

        # test model
        test_results = util_model.evaluate(model=model, data_loader=data_loaders['test'], device=device)
        print(test_results)

        # predictions
        preds, targs, times = util_model.predict(model=model, data_loader=data_loaders['test'], device=device)
        for pred in preds:
            predictions.append(pred)
        for targ in targs:
            targets.append(targ)
        for time in times:
            surv_times.append(time)

        # update report
        try:
            results_frame = pd.read_excel(report_file)
        except FileNotFoundError:
            results_frame = pd.DataFrame()

        new_data = pd.DataFrame([test_results])  # convert the new results to a dataframe
        new_data['Backbone'] = backbone_name     # add a new column with the model name
        new_data['Mode'] = mode                  # add a new column with the model name

        # concatenate the new data with the existing DataFrame
        results_frame = pd.concat([results_frame, new_data], ignore_index=True)

        # save results
        results_frame.to_excel(report_file, index=False)

    # compute metrics
    results = util_model.compute_performance(predictions, targets, surv_times)
    print("10 folds results: ", results)