# import torch
# from torchvision import models
#
# # Import your HIFD2 and BasicConv classes here
# from RFM_insect import BasicConv
# from RFM_insect import HIFD2
#
# # Load backbone as in your main.py
# backbone = models.resnet18(pretrained=False)
#
# # Use the same feature size and dataset as in your main.py
# feature_size = 512
# dataset = 'Insect'  # Or whatever value args.dataset would be
#
# # Instantiate your model
# model = HIFD2(backbone, feature_size, num_ftrs, dataset)
#
# # Count total parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params:,}")

# EffNetV2-S
# import torch
# from torchvision.models import efficientnet_v2_s
#
# # Import your custom modules
# from RFM_EffNet1 import BasicConv  # Imported but not used directly here
# from RFM_EffNet1 import HIFD2
#
# # Load EfficientNetV2-S backbone
# backbone = efficientnet_v2_s(weights=None)  # Set weights='IMAGENET1K_V1' for pretrained
#
# # Set output feature size from backbone
# num_ftrs = 320 # 1280  # EfficientNetV2-S outputs 1280 features
#
# # Set bottleneck feature size and dataset name
# feature_size = 128 # 640
# dataset = 'Insect'
#
# # Instantiate your model
# model = HIFD2(backbone, feature_size, num_ftrs, dataset)
#
# # Count total parameters
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# print(f"Total parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")
# print(f"Backbone output features (num_ftrs): {num_ftrs}")

# # EfficientNet_b0
# import torch
# from torchvision.models import efficientnet_b0
#
# def count_model_params(backbone, num_ftrs, feature_size, dataset='Insect'):
#     from RFM_EffNet1 import HIFD2  # Adjust import path if needed
#
#     model = HIFD2(backbone, feature_size, num_ftrs, dataset)
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     print(f"Backbone: {type(backbone).__name__}")
#     print(f"num_ftrs: {num_ftrs}, feature_size: {feature_size}")
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print("-" * 50)
#
# # EfficientNet-B0
# backbone_b0 = efficientnet_b0(weights=None)
# num_ftrs_b0 = 1280  # Output channels from EfficientNet-B0
# feature_size_b0 = 320
# count_model_params(backbone_b0, num_ftrs_b0, feature_size_b0)

# MobileNetV3-Small
from torchvision.models import mobilenet_v3_small
from RFM_EffNet1 import HIFD2

backbone = mobilenet_v3_small(weights=None)
num_ftrs = 576  # MobileNetV3-Small output
feature_size = 128

model = HIFD2(backbone, feature_size, num_ftrs, dataset='Insect')

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

from torchvision.models import mobilenet_v3_large
from RFM_EffNet1 import HIFD2

# # Load MobileNetV3 Large without pretrained weights
# backbone = mobilenet_v3_large(weights=None)
#
# # MobileNetV3 Large outputs 960 features
# num_ftrs = 960
# feature_size = 128
#
# # Initialize your model
# model = HIFD2(backbone, feature_size, num_ftrs, dataset='Insect')
#
# # Count total parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params:,}")