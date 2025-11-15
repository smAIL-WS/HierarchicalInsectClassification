"""
Main training script for hierarchical insect classification.
Uses centralized configuration from config.py.
"""
import random
import numpy as np
import timm
import torch
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import argparse
from torch.optim import lr_scheduler
from datetime import datetime
from multiprocessing import Manager
import signal
import sys

import config
from hierarchical_model import HIFD2, BackboneWrapper
from tree_loss import TreeLoss
from insect_dataset_loader import InsectDataset, DynamicTransform
from train_test import train, test


def arg_parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hierarchical Insect Classification Training')
    parser.add_argument('--worker', default=config.DEFAULT_NUM_WORKERS, type=int, 
                       help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=config.DEFAULT_SEED, 
                       help='random seed')
    parser.add_argument('--epoch', type=int, default=config.DEFAULT_NUM_EPOCHS, 
                       help='number of training epochs')
    parser.add_argument('--batch', type=int, default=config.DEFAULT_BATCH_SIZE, 
                       help='batch size')
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME, 
                       help='dataset name')
    parser.add_argument('--img_size', type=int, default=config.PROGRESSIVE_RESIZE_SCHEDULE['initial'],
                       help='initial image size')
    parser.add_argument('--lr_adjt', type=str, default='Cos', 
                       choices=['Cos', 'Step', 'Fixed'],
                       help='learning rate schedule: Cos, Step, or Fixed')
    parser.add_argument('--device', nargs='+', default=['0'], 
                       help='GPU IDs for training')
    parser.add_argument('--backbone', type=str, default='mobilenetv3_small', 
                       choices=list(config.BACKBONE_CONFIGS.keys()),
                       help='backbone model name')
    parser.add_argument('--use_pretrained', action='store_true', 
                       help='use custom pretrained weights for backbone')
    
    return parser.parse_args()

# Global epoch tracker
current_epoch = 0
manager = Manager()
epoch_tracker = manager.dict()
epoch_tracker["current"] = 0


def worker_init_fn(worker_id, epoch_tracker):
    """
        Initialize worker with proper random seeding and shared state.

    Args:
        worker_id: Worker ID assigned by PyTorch
        epoch_tracker: Shared dictionary containing current epoch number
    """

    # Set different seeds for each worker to ensure diverse data augmentation
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # Update the dataset's transform with current epoch
    # (This happens after the worker is forked, so each worker gets the epoch info)
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset.transform, "set_epoch"):
        dataset.transform.set_epoch(epoch_tracker["current"])

def get_transform(image_size):
    """
    Get training data augmentation transform.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    aug = config.TRAIN_AUGMENTATION
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(
            image_size - aug['crop_offset'], 
            scale=aug['resize_crop_scale']
        ),
        transforms.RandomHorizontalFlip() if aug['horizontal_flip'] else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(**aug['color_jitter']),
        transforms.RandomRotation(degrees=aug['rotation_degrees']),
        transforms.RandomAffine(degrees=0, translate=aug['affine_translate']),
        transforms.ToTensor(),
        transforms.Normalize(aug['normalize_mean'], aug['normalize_std']),
    ])


def get_test_transform(image_size):
    """
    Get test/validation data transform.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    aug = config.TEST_AUGMENTATION
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size - aug['crop_offset']),
        transforms.ToTensor(),
        transforms.Normalize(aug['normalize_mean'], aug['normalize_std']),
    ])


def setup_backbone(backbone_name, use_pretrained=False):
    """
    Setup backbone model with appropriate configuration.
    
    Args:
        backbone_name: Name of backbone architecture
        use_pretrained: Whether to load pretrained weights
        
    Returns:
        Tuple of (wrapped_backbone, num_ftrs, feature_size)
    """
    backbone_name = backbone_name.lower()
    
    if backbone_name not in config.BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported backbone: {backbone_name}. "
                        f"Choose from {list(config.BACKBONE_CONFIGS.keys())}")
    
    backbone_config = config.BACKBONE_CONFIGS[backbone_name]
    num_ftrs = backbone_config['num_ftrs']
    feature_sizes = backbone_config['feature_sizes']
    
    # Create backbone
    if backbone_name == 'resnet18':
        raw_backbone = models.resnet18(pretrained=False)
    elif backbone_name == 'resnet34':
        raw_backbone = models.resnet34(pretrained=False)
    elif backbone_name == 'resnet50':
        raw_backbone = models.resnet50(pretrained=False)
    elif backbone_name == 'resnext101':
        raw_backbone = models.resnext101_32x8d(pretrained=False)
    elif backbone_name == 'efficientnetv2_s':
        raw_backbone = timm.create_model('efficientnetv2_s', pretrained=False)
    elif backbone_name == 'mobilenetv3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        raw_backbone = mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(f"Backbone {backbone_name} not implemented")
    
    # # Load pretrained weights if requested
    # if use_pretrained and backbone_name in config.PRETRAINED_WEIGHTS:
    #     pretrained_path = config.PRETRAINED_WEIGHTS[backbone_name]
    #     if pretrained_path.exists():
    #         print(f"Loading pretrained weights from {pretrained_path}")
    #         raw_backbone.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    #     else:
    #         print(f"Warning: Pretrained weights not found at {pretrained_path}")
    
    # Wrap backbone for standardized output
    backbone = BackboneWrapper(raw_backbone)
    
    return backbone, num_ftrs, feature_sizes

def signal_handler(sig, frame):
    """Handle CTRL+C gracefully."""
    print('\n\n[INTERRUPT] Received CTRL+C - Cleaning up...')
    print('Killing worker processes...')
    sys.exit(0)

def main():
    """Main training function."""
    # Register signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    args = arg_parse()
    
    # Print configuration
    config.print_config()
    print(f"\nTraining Arguments:")
    print(f"  Epochs: {args.epoch}")
    print(f"  Batch size: {args.batch}")
    print(f"  Initial image size: {args.img_size}")
    print(f"  Device: {args.device}")
    print(f"  LR schedule: {args.lr_adjt}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Use pretrained: {args.use_pretrained}")
    print()
    
    # Setup data transforms
    image_size = args.img_size
    # transform_train = get_transform(image_size)
    # transform_test = get_test_transform(image_size)
    
    # Load datasets
    train_list = config.get_train_list_path()
    test_list = config.get_test_list_path()

    # === DynamicTransform ===
    # Instantiate dynamic transform with schedule from config
    dynamic_transform = DynamicTransform(
        schedule=config.PROGRESSIVE_RESIZE_SCHEDULE,
        aug_config=config.TRAIN_AUGMENTATION,
        epoch_tracker=epoch_tracker
    )

    # Apply dynamic transform to training dataset
    trainset = InsectDataset(train_list, input_transform=dynamic_transform)

    # Keep test transform logic unchanged
    transform_test = get_test_transform(image_size)
    testset = InsectDataset(test_list, input_transform=transform_test)

    # trainset = InsectDataset(train_list, transform_train)
    # testset = InsectDataset(test_list, transform_test)
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=args.batch, 
        shuffle=True, 
        num_workers=args.worker,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, epoch_tracker),
        drop_last = False
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=args.worker // 2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False
    )
    
    # Setup device
    device = torch.device(f"cuda:{args.device[0]}")
    
    # Setup backbone
    backbone, num_ftrs, feature_sizes = setup_backbone(args.backbone, args.use_pretrained)
    
    # Create model
    net = HIFD2(
        model=backbone,
        backbone_name=args.backbone,
        dataset=args.dataset
    )
    net.to(device)
    
    # Setup loss function
    tree_loss = TreeLoss(
        config.HIERARCHY, 
        device, 
        config.RUN_FOLDER,
        alpha=config.TREE_LOSS_ALPHA,
        invert=config.TREE_LOSS_INVERT,
        sample_count_file=config.NODE_SAMPLE_COUNTS_FILE,
        beta=config.TREE_LOSS_BETA
    )
    
    # Setup optimizer
    optimizer = optim.SGD([
        {'params': net.classifier_pf4.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.classifier_pf3.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.classifier_pf2.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.classifier_pf1.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.classifier_leaf.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.fc_pf4.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.fc_pf3.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.fc_pf2.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.fc_pf1.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.fc_leaf_class.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.conv_block_pf4.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.conv_block_pf3.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.conv_block_pf2.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.conv_block_pf1.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.conv_block_leaf_class.parameters(), 'lr': config.LR_CLASSIFIER},
        {'params': net.backbone.parameters(), 'lr': config.LR_BACKBONE}
    ], momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    # Setup learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=config.LR_SCHEDULER_STEP_SIZE, 
        gamma=config.LR_SCHEDULER_GAMMA
    )
    
    # Generate model save name
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_name = f"{args.dataset}_{args.epoch}_{args.img_size}_bz{args.batch}_{args.backbone}_{args.lr_adjt}_{now}"
    
    # Train model
    train(
        args.epoch, 
        net, 
        trainloader, 
        testloader, 
        optimizer, 
        scheduler, 
        args.lr_adjt, 
        args.dataset, 
        tree_loss,
        device, 
        args.device, 
        save_name, 
        trainset, 
        config.HIERARCHY, 
        get_transform, 
        get_test_transform, 
        run_folder=config.RUN_FOLDER,
        epoch_tracker=epoch_tracker
    )
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    test(
        net, 
        testloader, 
        tree_loss, 
        device, 
        args.dataset, 
        trainset,
        config.HIERARCHY, 
        get_test_transform, 
        image_size, 
        run_folder=config.RUN_FOLDER
    )


if __name__ == '__main__':
    main()
