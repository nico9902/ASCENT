# import libraries
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import pandas as pd
import numpy as np
import collections
import yaml

import torch
import argparse
import torchvision.transforms as transforms

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model
import src.model.networks as networks
import src.data.datasets as dataset
import src.model.collate as collate

# empty the cache
torch.cuda.empty_cache()

# experiment name
parser = argparse.ArgumentParser(description="Experiment Configuration")
parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")
parser.add_argument("--cfg_file", type=str, default="./configs/config_MST.yaml", help="Path to the configuration file")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
args = parser.parse_args()
exp_name = args.exp_name

# Update configuration file path and seed
with open(args.cfg_file) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
util_general.seed_all(args.seed)

# parameters
backbone_output_size = cfg['backbone']['backbone_output_size']
mode = cfg['model']['mode']
cv = cfg['data']['cv']
fold_start = cfg['data']['fold_start']
augmentation = cfg['data']['augmentation']
fold_list = list(range(cv))
pretrained = True

# deephit parameters
input_dims = cfg['deephit']['input_dims']
network_settings = cfg['deephit']['network_settings']
num_Event = util_general.check_na(input_dims['num_Event'])
num_Category = util_general.check_na(input_dims['num_Category'])
network_settings['active_fn'] = util_general.check_active_fn(network_settings['active_fn'])
alpha = cfg['deephit']['alpha']
beta = cfg['deephit']['beta']

# device
device = torch.device("cuda:0")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']


# main loop
if __name__=="__main__":
    # loop over the folds
    results = collections.defaultdict(lambda: [])
    acc_cols = []
    acc_class_cols = collections.defaultdict(lambda: [])
    predictions = []
    targets = []
    surv_times = []
    for fold in range(fold_start, cv):
        # create datasets
        fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.csv' % step), delimiter=",") for step in ['train', 'val', 'test']}
        datasets = {step: dataset.ImgDataset(patientIDs=fold_data[step]['PatientID'].tolist(), cfg_data=cfg['data'], time=fold_data[step]['Time'].tolist(), label=fold_data[step]['Label'].tolist(), augmentation=augmentation, step=step, num_Event=num_Event, num_Category=num_Category) for step in ['train', 'val', 'test']}
        print(f"Fold {fold}: Train {len(datasets['train'])}, Val {len(datasets['val'])}, Test {len(datasets['test'])}")

        # define data transform as a composition of Convert and Normalize
        # here, Normalize implements standardization using the computed mu and std of the training set
        # add Augmentation in training step if required
        mu = datasets['train'].mu
        std = datasets['train'].std
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
        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['model']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker, collate_fn=collate.MyCollator(stage='train', mask_percentage=cfg['model']['mask_perc']), drop_last=True),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['model']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker, collate_fn=collate.MyCollator(stage='val', mask_percentage=cfg['model']['mask_perc']), drop_last=False),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['model']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker, collate_fn=collate.MyCollator(stage='test', mask_percentage=cfg['model']['mask_perc']), drop_last=False)}

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
            print("Soft Attention Model", flush=True)
            model = networks.SoftAttentionModel(backbone=backbone, backbone_output_size=backbone_output_size, mlp_layer=cfg['model']['mlp'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings)
            # load pretrained weights (if needed)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{cfg['pre_exp_name']}.pt"
                pretrained_dict = torch.load(pretrained_model_weights, weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        elif mode == "mean":
            print("Average pooling", flush=True)
            model = networks.SoftAttentionModel(backbone=backbone, backbone_output_size=backbone_output_size, mlp_layer=cfg['model']['mlp'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings, fusion_criterion="mean")
            # load pretrained weights (if needed)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{backbone_name}.pt"
                pretrained_dict = torch.load(pretrained_model_weights, weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        elif mode == "kron":
            model = networks.SoftAttentionModel(backbone=backbone, backbone_output_size=backbone_output_size, mlp_layer=cfg['model']['mlp'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings, fusion_criterion="kronecker")
        elif mode== "trans_encoder":
            print("Self Attention", flush=True)
            model = networks.TransEncoderModel(backbone=backbone, embed_size=backbone_output_size, num_layers=cfg['model']['num_layers'], num_heads=cfg['model']['num_heads'], mlp_layer=cfg['model']['mlp'], conv_layer=cfg['model']['conv'], return_attention_weights=False, input_dims=input_dims, network_settings=network_settings)
            # load pretrained weights (if needed)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{backbone_name}.pt"
                pretrained_dict = torch.load(pretrained_model_weights, weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        elif mode == "majority_voting":
            model = networks.VotingModel(backbone=backbone, backbone_output_size=backbone_output_size, num_classes=cfg['data']['num_classes'])
        # 3D networks
        elif mode == "ResNet3D":
            print("ResNet3D", flush=True)
            model = networks.ResNet183D(cfg, pretrained=True, input_dims=input_dims, network_settings=network_settings, embedding_size=backbone_output_size)
            # load pretrained weights (if needed)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{cfg['pre_exp_name']}.pt"
                pretrained_dict = torch.load(pretrained_model_weights, weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        elif mode in ["densenet121", "densenet161", "densenet169", "densenet201"]:
            print("DenseNet3D", flush=True)
            model = networks.DenseNet3D(cfg, backbone_name=mode, input_dims=input_dims, network_settings=network_settings, embedding_size=backbone_output_size)
            # load pretrained weights (if needed)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{cfg['pre_exp_name']}.pt"
                pretrained_dict = torch.load(pretrained_model_weights, weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        elif mode == "DINOv2":
            print("MST 3D", flush=True)
            model = networks.DINOv2(cfg, input_dims=input_dims, network_settings=network_settings, embedding_size=backbone_output_size)
            if pretrained:
                pretrained_model_weights = f"experiments/model_dir/{cfg['pre_exp_name']}/{fold}/{cfg['pre_exp_name']}.pt"
                pretrained_dict = torch.load(pretrained_model_weights, weights_only=False).state_dict()
                remove_prefix = 'module.'
                pretrained_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in pretrained_dict.items()}
                model.load_state_dict(pretrained_dict)
        else:
            raise ValueError("Il valore di mode non è valido.")

        # test model
        model = model.to(device)
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

        # # update report
        # try:
        #     results_frame = pd.read_excel('CLARO_report.xlsx')
        # except FileNotFoundError:
        #     results_frame = pd.DataFrame()

        # new_data = pd.DataFrame([test_results])  # convert the new results to a dataframe
        # new_data['Backbone'] = backbone_name     # add a new column with the model name
        # new_data['Mode'] = mode                  # add a new column with the model name

        # # concatenate the new data with the existing DataFrame
        # results_frame = pd.concat([results_frame, new_data], ignore_index=True)

        # # save results
        # results_frame.to_excel('CLARO_report.xlsx', index=False)

    # compute metrics
    results = util_model.compute_performance(predictions, targets, surv_times)
    print("10 folds results: ", results)

    # save predictions, targets and survival times
    predictions = np.array(predictions)
    targets = np.array(targets)
    surv_times = np.array(surv_times)
    os.makedirs(f"predictions/{exp_name}", exist_ok=True)
    np.save(f"predictions/{exp_name}/predictions.npy", predictions)
    np.save(f"predictions/{exp_name}/targets.npy", targets)
    np.save(f"predictions/{exp_name}/surv_times.npy", surv_times)