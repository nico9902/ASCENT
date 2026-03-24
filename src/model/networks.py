# import libraries
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import src.utils.util_network as util_network
from src.utils import util_model as util_model
from src.model import densenet as densenet

from MST.mst.models.dino import DinoV2ClassifierSlice

class DINOv2(nn.Module):
    def __init__(self, cfg, input_dims=None, network_settings=None, embedding_size=256):
        super(DINOv2, self).__init__()
        self.backbone3D = DinoV2ClassifierSlice(in_ch=1, out_ch=2)
        # if cfg['backbone']['pretrained']:
        #     checkpoint = torch.load("/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/deep-lung/MedViT/MedViT_small_im1k.pth", weights_only=False)  #("/Users/domenicopaolo/Documents/PhD AI/Projects/deep-lung/MST/MST_LIDC.ckpt", weights_only=False, map_location='cpu')
        #     self.backbone3D.load_state_dict(checkpoint["state_dict"])
        
        util_model.set_parameter_requires_grad(self.backbone3D, freeze=cfg['backbone']['freeze'], half_freeze=cfg['backbone']['half_freeze'], unfreeze_last=cfg['backbone']['unfreeze_last'])
        self.backbone3D = self.backbone3D.to(torch.device("cuda:0"))
        self.input_dims = input_dims
        self.network_settings = network_settings

        num_ftrs = self.backbone3D.linear.in_features
        self.backbone3D.linear = nn.Linear(in_features=num_ftrs, out_features=embedding_size)

        # Output layer
        self.out_layer = DeepHit(self.input_dims, self.network_settings)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.backbone3D(x)
        x = self.out_layer(x)
        return x


# DenseNet3D model
class DenseNet3D(nn.Module):
    def __init__(self, cfg, backbone_name, input_dims=None, network_settings=None, embedding_size=256):
        super(DenseNet3D, self).__init__()
        self.backbone3D = densenet.getDenseNet(backbone_name)
        util_model.set_parameter_requires_grad(self.backbone3D, freeze=cfg['backbone']['freeze'], half_freeze=cfg['backbone']['half_freeze'], unfreeze_last=cfg['backbone']['unfreeze_last'])
        self.input_dims = input_dims
        self.network_settings = network_settings
        
        # Modify the final fully connected layer to match the desired output size, if necessary
        num_ftrs = self.backbone3D.classifier.in_features
        self.backbone3D.classifier = nn.Linear(num_ftrs, embedding_size)

        # Output layer
        self.out_layer = DeepHit(self.input_dims, self.network_settings)
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.backbone3D(x)
        x = self.out_layer(x)
        return x
    
# ResNet18_3D model
class ResNet183D(nn.Module):
    """
        ResNet18_3D model for 3D image classification.
    """
    def __init__(self, cfg, pretrained=False, input_dims=None, network_settings=None, embedding_size=256):
        super(ResNet183D, self).__init__()
        self.backbone3D = models.video.r3d_18(pretrained=pretrained)
        self.backbone3D.stem[0].in_channels = 1
        self.backbone3D.stem[0].weight = torch.nn.Parameter(self.backbone3D.stem[0].weight.sum(1, keepdim=True))
        util_model.set_parameter_requires_grad(self.backbone3D, freeze=cfg['backbone']['freeze'], half_freeze=cfg['backbone']['half_freeze'], unfreeze_last=cfg['backbone']['unfreeze_last'])
        self.input_dims = input_dims
        self.network_settings = network_settings
        
        # Modify the final fully connected layer to match the desired output size, if necessary
        num_ftrs = self.backbone3D.fc.in_features
        self.backbone3D.fc = nn.Linear(num_ftrs, embedding_size) 
        
        # output layer
        self.out_layer = DeepHit(self.input_dims, self.network_settings)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.backbone3D(x)
        x = self.out_layer(x)
        return x

# soft attention module
class SoftAttention(nn.Module):
    """
        Soft attention mechanism to compute the weighted mean of the input tensor.
    """
    def __init__(self, input_size, mlp=False):
        super(SoftAttention, self).__init__()
        self.mlp = mlp
        if self.mlp:
            # if MLP is selected, use an MLP as classification layer
            self.classifier = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # use a fully connected layer
            self.classifier = nn.Linear(input_size, 1)

    def forward(self, inputs): 
        # get batch size and sequence length from inputs
        batch_size, depth, _ = inputs.size()

        # create a mask to ignore padded elements
        mask = (inputs.sum(dim=2) != 0).float()
        #mask = torch.all(inputs != 0, dim=2).float()

        # get attention scores
        scores = self.classifier(inputs).view(batch_size, depth)

        # apply softmax to get attention weights
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # # normalize scores
        # scores = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)

        # apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # Apply temperature scaling with temperature=0.1
        #attention_weights = util_network.safe_softmax(scores)

        # compute the weighted mean
        attention_weights = attention_weights.unsqueeze(2)
        weighted_sum = torch.bmm(inputs.transpose(1, 2), attention_weights).squeeze(2)

        return weighted_sum, attention_weights

# DeepHit model risk estimator network
class DeepHit(nn.Module):
    """
        DeepHit model for survival analysis.
        It is a multi-task learning model that estimates the probability of each event for each time step.
    """
    def __init__(self, input_dims, network_settings):
        super(DeepHit, self).__init__()
        # input dimensions
        self.x_dim              = input_dims['x_dim']
        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']

        # network hyper-parameters
        self.h_dim_shared       = network_settings['h_dim_shared']
        self.h_dim_CS           = network_settings['h_dim_CS']
        self.num_layers_shared  = network_settings['num_layers_shared']
        self.num_layers_CS      = network_settings['num_layers_CS']
        self.active_fn          = network_settings['active_fn']
        self.p_dropout          = network_settings['dropout']
        self.limit = None

        # dropout
        self.dropout = nn.Dropout(p=self.p_dropout)

        # shared network
        self.shared_net = util_network.create_FCNet(self.x_dim, self.num_layers_shared, self.h_dim_shared, self.active_fn, self.h_dim_shared, self.active_fn, self.limit, self.dropout)
        
        # cause-specific network
        self.cs_net = util_network.create_FCNet(self.x_dim + self.h_dim_shared, (self.num_layers_CS), self.h_dim_CS, self.active_fn, self.h_dim_CS, self.active_fn, self.limit, self.dropout)

        # output layer
        self.out_fc = nn.Linear(self.num_Event*self.h_dim_CS, self.num_Event*self.num_Category)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.m = nn.Softmax(dim=1)
    
    def forward(self, x):
        # shared network
        shared_out = self.shared_net(x)

        # residual connection
        shared_out = torch.cat([x, shared_out], dim=1)

        # (num_layers_CS) layers for cause-specific (num_Event subNets)
        out = []
        for _ in range(self.num_Event):
            cs_out = self.cs_net(shared_out)
            out.append(cs_out)
        out = torch.stack(out, dim=1) # stack referenced on subject
        out = torch.reshape(out, (-1, self.num_Event*self.h_dim_CS))
        out = self.dropout(out)

        out = self.out_fc(out)
        out = self.m(out)
        out = torch.reshape(out, (-1, self.num_Event, self.num_Category))

        return out

# soft attention model
class SoftAttentionModel(nn.Module):
    """
        Soft attention model for survival analysis.
        It computes the weighted mean of the input tensor using a soft attention mechanism (or a mean).
        Survival predictions are made using the weighted mean (or the mean).
    """
    def __init__(self, backbone, backbone_output_size, mlp_layer=False, return_attention_weights=False, input_dims=None, network_settings=None, fusion_criterion='soft_attention'):
        super(SoftAttentionModel, self).__init__()
        self.backbone = backbone
        self.backbone_output_size = backbone_output_size
        self.mlp_layer = mlp_layer
        self.attention_layer = SoftAttention(input_size=self.backbone_output_size, mlp=self.mlp_layer)
        self.return_attention_weights = return_attention_weights
        self.input_dims = input_dims
        self.network_settings = network_settings
        self.fusion_criterion = fusion_criterion
        
        # output layer
        self.out_layer = DeepHit(self.input_dims, self.network_settings)
        
    def forward(self, x):
        # slices mask (to ignore padded slices)
        mask = ~(x==0).all(dim=-1) & ~(x==0).all(dim=-2)
        mask = (mask.float().sum(dim=(2,3)) != 0).float()

        # transform the tensor to the desired size 
        inputs = x.view(-1, *x.shape[2:])                                               # e.g. [batch_size, num_slices, num_channels, H, W] -> [batch_size * num_slices, num_channels, H, W]
    
        # get backbone outputs
        backbone_outputs = self.backbone(inputs)                                        # e.g. [batch_size * num_slices, embedding_size]

        # transform the tensor to the desired size
        backbone_outputs = backbone_outputs.view(*x.shape[:2], -1)                      # e.g. [batch_size * num_slices, embedding_size] -> [batch_size, num_slices, embedding_size]
    
        # delete padded slices
        backbone_outputs = backbone_outputs * mask.unsqueeze(-1)                        # e.g. [batch_size, num_slices, embedding_size]

        if self.fusion_criterion == "mean":
            # compute slices-level soft attention
            surv_features = backbone_outputs.mean(dim=1)                               # e.g. [batch_size, num_slices, embedding_size] -> [batch_size, embedding_size]
        elif self.fusion_criterion == "soft_attention":
            # compute slices-level soft attention
            surv_features, attention_weights = self.attention_layer(backbone_outputs)  # e.g. [batch_size, num_slices, embedding_size] -> [batch_size, embedding_size]
            attention_weights = attention_weights.squeeze()
        else:
            raise ValueError("Invalid fusion criterion")

        # output layer
        logits = self.out_layer(surv_features)      # if regression, e.g. [batch_size, embedding_size] -> [batch_size, num_Event, num_Category]
                                                    
        if self.return_attention_weights:
            return logits, attention_weights
        else:
            return logits

# transformer encoder model
class TransEncoderModel(nn.Module):
    """
        Transformer encoder model for survival risk estimation.
        It uses a transformer encoder to compute the class token of the input tensor.
        Survival predictions are made using the class token.
    """
    def __init__(self, backbone, embed_size, num_layers, num_heads, mlp_layer=False, conv_layer=False, return_attention_weights=False, input_dims=None, network_settings=None):
        super(TransEncoderModel, self).__init__()
        self.backbone = backbone
        self.backbone_output_size = embed_size
        self.flatten = nn.Flatten()
        self.mlp_layer = mlp_layer
        self.conv_layer = conv_layer
        self.layers = nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    embed_size,
                    num_heads,
                    batch_first=True,
                    norm_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.return_attention_weights = return_attention_weights
        self.input_dims = input_dims
        self.network_settings = network_settings

        # output layer
        self.out_layer = DeepHit(self.input_dims, self.network_settings)
        
    def forward(self, x):
        # mask slices (to ignore padded slices)
        mask_w = ~(x==0).all(dim=-1)
        mask_h = ~(x==0).all(dim=-2)
        mask = mask_w & mask_h
        mask = (mask.float().sum(dim=(2,3)) != 0).float()

        # transform the tensor to the desired size 
        inputs = x.view(-1, *x.shape[2:])  # e.g. [B, D, C, H, W] -> [B*D, C, H, W]

        # get backbone outputs
        backbone_outputs = self.backbone(inputs)  # e.g. [B*D, E]

        # transform the tensor to the desired size
        backbone_outputs = backbone_outputs.view(*x.shape[:2], -1)  # e.g. [B*D, E] -> [B, D, E]
        
        # delete padded slices
        backbone_outputs = backbone_outputs * mask.unsqueeze(-1)  # e.g. [B, D, E]

        # # add cls token to the batch 
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, D, E)
        backbone_outputs = torch.cat((cls_tokens, backbone_outputs), dim=1)  # (B, D, E)

        # compute soft attention
        attended_features = backbone_outputs
        for layer in self.layers:
            attended_features = layer(backbone_outputs)

        # Extract the cls token output 
        attended_features = attended_features[:, 0]  # (B, E)

        # classification
        logits = self.out_layer(attended_features)  # if regression, e.g. [B, num_Event, num_Category]
                                                    
        return logits

# voting model
class VotingModel(nn.Module):
    """
        Voting model for survival classification.
    """
    def __init__(self, backbone, backbone_output_size, num_classes):
        super(VotingModel, self).__init__()
        self.backbone = backbone
        self.backbone_output_size = backbone_output_size
        self.num_classes = num_classes

        # classification layer
        self.fc1 = nn.Linear(self.backbone_output_size, self.num_classes)

    def forward(self, x):
        # mask slices
        mask_w = ~(x==0).all(dim=-1)
        mask_h = ~(x==0).all(dim=-2)
        mask = mask_w & mask_h
        mask = (mask.float().sum(dim=(2,3)) != 0).float()
        
        # transform the tensor to the desired size 
        inputs = x.view(-1, *x.shape[2:])  # e.g. [B, D, C, H, W] -> [B*D, C, H, W]

        # get backbone outputs
        backbone_outputs = self.backbone(inputs)  # e.g. [B*D, E]

        # transform the tensor to the desired size
        backbone_outputs = backbone_outputs.view(*x.shape[:2], -1)  # e.g. [B*D, E] -> [B, D, E]
        
        # delete padded slices
        backbone_outputs = backbone_outputs * mask.unsqueeze(-1)  # e.g. [B, D, E]

        # classification
        logits = self.fc1(backbone_outputs)                                             # e.g. [B, D, E] -> [B, D, num_classes]

        # Apply majority voting
        logits_majority_voted = util_network.majority_voting(logits=logits, mask=mask.bool())  # e.g. [B, num_slice_non_zero, num_classes]

        return logits_majority_voted
