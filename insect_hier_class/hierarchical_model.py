"""
Hierarchical Feature Difference model (HIFD2) for insect classification.
Supports multiple backbone architectures with hierarchical output heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BackboneWrapper(nn.Module):
    """
    Wrapper to standardize output from different backbone architectures.
    Removes classifier heads and provides feature extraction interface.
    """
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Remove classifier if it exists (e.g., MobileNetV3, EfficientNet)
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
    
    def forward(self, x):
        """Extract features from backbone."""
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x)
        elif hasattr(self.backbone, 'features'):
            return self.backbone.features(x)
        else:
            return self.backbone(x)


class BasicConv(nn.Module):
    """
    Basic convolutional block with BatchNorm, ReLU, and optional Dropout.
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, relu=True, bn=True, bias=False, 
                 dropout_rate=None):
        super(BasicConv, self).__init__()
        
        if dropout_rate is None:
            dropout_rate = config.DROPOUT_RATE_CONV
        
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes, out_planes, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
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


def create_conv_block(num_ftrs, feature_size, dropout_rate=None):
    """
    Factory function to create convolutional refinement blocks.
    
    Args:
        num_ftrs: Number of input features
        feature_size: Intermediate feature dimension
        dropout_rate: Dropout rate (uses config default if None)
        
    Returns:
        Sequential conv block
    """
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE_CONV
    
    return nn.Sequential(
        BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, 
                 relu=True, dropout_rate=dropout_rate),
        BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, 
                 relu=True, dropout_rate=dropout_rate)
    )


def create_fc_block(num_ftrs, feature_size, output_dim=None, dropout_rate=None):
    """
    Factory function to create fully connected blocks.
    
    Args:
        num_ftrs: Number of input features
        feature_size: Intermediate feature dimension
        output_dim: Output dimension (uses config default if None)
        dropout_rate: Dropout rate (uses config default if None)
        
    Returns:
        Sequential FC block
    """
    if output_dim is None:
        output_dim = config.FC_HIDDEN_DIM
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE_FC
    
    return nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, feature_size),
        nn.BatchNorm1d(feature_size),
        nn.ELU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(feature_size, output_dim)
    )


def create_classifier(input_dim, num_classes, use_sigmoid=True):
    """
    Factory function to create classification heads.

    Args:
        input_dim: Input dimension
        num_classes: Number of output classes
        use_sigmoid: Whether to apply sigmoid activation (DEPRECATED - use raw logits)

    Returns:
        Sequential classifier (outputs raw logits)
    """
    # Return only linear layer - no activation
    # This allows dual use for both BCE and CrossEntropy losses
    return nn.Linear(input_dim, num_classes)


class HIFD2(nn.Module):
    """
    Hierarchical Feature Difference model with multiple output heads
    for different hierarchy levels.
    """
    
    def __init__(self, model, backbone_name, dataset):
        """
        Args:
            model: Wrapped backbone model
            feature_size: Intermediate feature dimension
            num_ftrs: Number of features from backbone
            dataset: Dataset name (for compatibility)
        """
        super(HIFD2, self).__init__()

        # Get backbone-specific config
        backbone_cfg = config.BACKBONE_CONFIGS[backbone_name]
        num_ftrs = backbone_cfg['num_ftrs']
        sizes = backbone_cfg['feature_sizes']

        self.backbone = model
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        # Convolutional refinement blocks per level
        self.conv_block_pf4 = create_conv_block(num_ftrs, sizes['pf4'])
        self.conv_block_pf3 = create_conv_block(num_ftrs, sizes['pf3'])
        self.conv_block_pf2 = create_conv_block(num_ftrs, sizes['pf2'])
        self.conv_block_pf1 = create_conv_block(num_ftrs, sizes['pf1'])
        self.conv_block_leaf_class = create_conv_block(num_ftrs, sizes['leaf'])

        # Fully connected blocks per level
        self.fc_pf4 = create_fc_block(num_ftrs, sizes['pf4'])
        self.fc_pf3 = create_fc_block(num_ftrs, sizes['pf3'])
        self.fc_pf2 = create_fc_block(num_ftrs, sizes['pf2'])
        self.fc_pf1 = create_fc_block(num_ftrs, sizes['pf1'])
        self.fc_leaf_class = create_fc_block(num_ftrs, sizes['leaf'])

        # Classification heads (raw logits output for dual use)
        self.classifier_pf4 = create_classifier(
            config.FC_HIDDEN_DIM,
            config.NUM_CLASSES_PER_LEVEL['pf4'],
            use_sigmoid=False  # Output raw logits
        )
        self.classifier_pf3 = create_classifier(
            config.FC_HIDDEN_DIM,
            config.NUM_CLASSES_PER_LEVEL['pf3'],
            use_sigmoid=False
        )
        self.classifier_pf2 = create_classifier(
            config.FC_HIDDEN_DIM,
            config.NUM_CLASSES_PER_LEVEL['pf2'],
            use_sigmoid=False
        )
        self.classifier_pf1 = create_classifier(
            config.FC_HIDDEN_DIM,
            config.NUM_CLASSES_PER_LEVEL['pf1'],
            use_sigmoid=False
        )
        self.classifier_leaf = create_classifier(
            config.FC_HIDDEN_DIM,
            config.NUM_CLASSES_PER_LEVEL['leaf'],
            use_sigmoid=False
        )

        # Remove the separate leaf_sig and leaf_soft classifiers
        # Use single classifier with raw logits

    def forward(self, x):
        """
        Forward pass through hierarchical model.

        Args:
            x: Input images [batch_size, 3, H, W]

        Returns:
            Tuple of raw logits (pf4, pf3, pf2, pf1, leaf)
        """
        # Extract backbone features
        x = self.backbone(x)

        # Refine features for each hierarchy level
        x_pf4 = self.conv_block_pf4(x)
        x_pf3 = self.conv_block_pf3(x)
        x_pf2 = self.conv_block_pf2(x)
        x_pf1 = self.conv_block_pf1(x)
        x_leaf_class = self.conv_block_leaf_class(x)

        # Global average pooling and flatten
        x_pf4_fc = self.pooling(x_pf4).view(x_pf4.size(0), -1)
        x_pf3_fc = self.pooling(x_pf3).view(x_pf3.size(0), -1)
        x_pf2_fc = self.pooling(x_pf2).view(x_pf2.size(0), -1)
        x_pf1_fc = self.pooling(x_pf1).view(x_pf1.size(0), -1)
        x_leaf_class_fc = self.pooling(x_leaf_class).view(x_leaf_class.size(0), -1)

        # FC transformation
        x_pf4_fc = self.fc_pf4(x_pf4_fc)
        x_pf3_fc = self.fc_pf3(x_pf3_fc)
        x_pf2_fc = self.fc_pf2(x_pf2_fc)
        x_pf1_fc = self.fc_pf1(x_pf1_fc)
        x_leaf_class_fc = self.fc_leaf_class(x_leaf_class_fc)

        # Hierarchical feature aggregation with ReLU
        # Each level accumulates features from coarser levels
        y_pf4 = self.classifier_pf4(self.relu(x_pf4_fc))

        y_pf3 = self.classifier_pf3(self.relu(
            x_pf4_fc + x_pf3_fc
        ))

        y_pf2 = self.classifier_pf2(self.relu(
            x_pf4_fc + x_pf3_fc + x_pf2_fc
        ))

        y_pf1 = self.classifier_pf1(self.relu(
            x_pf4_fc + x_pf3_fc + x_pf2_fc + x_pf1_fc
        ))

        aggregated_features = self.relu(
            x_pf4_fc + x_pf3_fc + x_pf2_fc + x_pf1_fc + x_leaf_class_fc
        )

        y_leaf = self.classifier_leaf(aggregated_features)

        # Return raw logits - apply sigmoid/softmax in loss functions
        return y_pf4, y_pf3, y_pf2, y_pf1, y_leaf
