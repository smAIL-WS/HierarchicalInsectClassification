# Updated for 5 levels of hierarchy

import torch.nn as nn
import torch
import torch.nn.functional as F

class BackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Remove classifier if it exists (e.g., MobileNetV3)
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()

    def forward(self, x):
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x)
        elif hasattr(self.backbone, 'features'):
            return self.backbone.features(x)
        else:
            return self.backbone(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False, dropout_rate=0.2):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # Post-activation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class HIFD2(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, dataset):
        super(HIFD2, self).__init__()

        self.backbone = model
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = num_ftrs

        self.conv_block_pf4 = nn.Sequential( # alternative pf4 with depthwise separable convolution
            nn.Conv2d(self.num_ftrs, self.num_ftrs, kernel_size=3, stride=1, padding=1, groups=self.num_ftrs,
                      bias=False),  # Depthwise
            nn.BatchNorm2d(self.num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_ftrs, self.num_ftrs, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(self.num_ftrs),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        # self.conv_block_pf4 = nn.Sequential(
        #     BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True, dropout_rate=0.2),
        #     BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True, dropout_rate=0.2)
        # )

        self.conv_block_pf3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True, dropout_rate=0.2),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True, dropout_rate=0.2)
        )
        self.conv_block_pf2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True, dropout_rate=0.2),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True, dropout_rate=0.2)
        )
        self.conv_block_pf1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True, dropout_rate=0.2),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True, dropout_rate=0.2)
        )
        self.conv_block_leaf_class = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True, dropout_rate=0.2),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True, dropout_rate=0.2)
        )

        self.fc_pf4 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),  # Dropout added here
            nn.Linear(feature_size, 512)
        )

        self.fc_pf3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),  # Dropout added here
            nn.Linear(feature_size, 512)
        )

        self.fc_pf2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),  # Dropout added here
            nn.Linear(feature_size, 512)
        )

        self.fc_pf1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),  # Dropout added here
            nn.Linear(feature_size, 512)
        )

        self.fc_leaf_class = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),  # Dropout added here
            nn.Linear(feature_size, 512)
        )

        self.classifier_pf4 = nn.Sequential(
            nn.Linear(512, 2),  # Number of parent_folder_4 classes
            nn.Sigmoid()
        )
        self.classifier_pf3 = nn.Sequential(
            nn.Linear(512, 11),
            nn.Sigmoid()
        )
        self.classifier_pf2 = nn.Sequential(
            nn.Linear(512, 13),
            nn.Sigmoid()
        )
        self.classifier_pf1 = nn.Sequential(
            nn.Linear(512, 15),
            nn.Sigmoid()
        )
        self.classifier_leaf_sig = nn.Sequential(
            nn.Linear(512, 22),
            nn.Sigmoid()
        )
        self.classifier_leaf_soft = nn.Sequential(
            nn.Linear(512, 22)
        )

    def forward(self, x):
        x = self.backbone(x)
        # print("Backbone output shape:", x.shape)

        # print(x.shape)  # Should be [batch_size, num_ftrs, H, W]
        x_pf4 = self.conv_block_pf4(x)
        x_pf3 = self.conv_block_pf3(x)
        x_pf2 = self.conv_block_pf2(x)
        x_pf1 = self.conv_block_pf1(x)
        x_leaf_class = self.conv_block_leaf_class(x)

        x_pf4_fc = self.pooling(x_pf4)
        x_pf4_fc = x_pf4_fc.view(x_pf4_fc.size(0), -1)
        x_pf4_fc = self.fc_pf4(x_pf4_fc)

        x_pf3_fc = self.pooling(x_pf3)
        x_pf3_fc = x_pf3_fc.view(x_pf3_fc.size(0), -1)
        x_pf3_fc = self.fc_pf3(x_pf3_fc)

        x_pf2_fc = self.pooling(x_pf2)
        x_pf2_fc = x_pf2_fc.view(x_pf2_fc.size(0), -1)
        x_pf2_fc = self.fc_pf2(x_pf2_fc)

        x_pf1_fc = self.pooling(x_pf1)
        x_pf1_fc = x_pf1_fc.view(x_pf1_fc.size(0), -1)
        x_pf1_fc = self.fc_pf1(x_pf1_fc)

        x_leaf_class_fc = self.pooling(x_leaf_class)
        x_leaf_class_fc = x_leaf_class_fc.view(x_leaf_class_fc.size(0), -1)
        x_leaf_class_fc = self.fc_leaf_class(x_leaf_class_fc)

        y_pf4_sig = self.classifier_pf4(self.relu(x_pf4_fc))
        y_pf3_sig = self.classifier_pf3(self.relu(x_pf4_fc + x_pf3_fc))
        y_pf2_sig = self.classifier_pf2(self.relu(x_pf4_fc + x_pf3_fc + x_pf2_fc))
        y_pf1_sig = self.classifier_pf1(self.relu(x_pf4_fc + x_pf3_fc + x_pf2_fc + x_pf1_fc))
        y_leaf_class_sig = self.classifier_leaf_sig(
            self.relu(x_pf4_fc + x_pf3_fc + x_pf2_fc + x_pf1_fc + x_leaf_class_fc))
        y_leaf_class_soft = self.classifier_leaf_soft(
            self.relu(x_pf4_fc + x_pf3_fc + x_pf2_fc + x_pf1_fc + x_leaf_class_fc))

        return y_pf4_sig, y_pf3_sig, y_pf2_sig, y_pf1_sig, y_leaf_class_sig, y_leaf_class_soft