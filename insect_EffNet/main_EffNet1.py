from PIL import Image
import timm
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import mobilenet_v3_small
import torch.hub
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import Subset  # added to allow a subset of the dataset to be used for debugging
from datetime import datetime

from RFM_EffNet1 import HIFD2, BackboneWrapper

from tree_loss_insect_EffNet1 import TreeLoss
from dataset_insect_EffNet1 import InsectDataset  # InsectDataset2
from train_test_insect_EffNet1 import test, test_AP
from train_test_insect_EffNet1 import train


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--worker', default=8, type=int, help='number of workers')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth',
                        help='Path of trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--epoch', type=int, default=100, help='Epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--dataset', type=str, default='Insect', help='dataset name')
    parser.add_argument('--img_size', type=str, default='112', help='image size')
    parser.add_argument('--lr_adjt', type=str, default='Cos',
                        help='Learning rate schedule')  # Changed from 'Cos' to 'Fixed' for debugging
    parser.add_argument('--device', nargs='+', default='0', help='GPU IDs for DP training')
    parser.add_argument('--backbone', type=str, default='mobilenetv3_small', help='Backbone model name')
    parser.add_argument('--use_pretrained', action='store_true', help='Use custom pretrained weights')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parse()
    print('==> epoch: ', args.epoch)
    print('==> batch: ', args.batch)
    print('==> dataset: ', args.dataset)
    print('==> img_size: ', args.img_size)
    print('==> device: ', args.device)
    print('==> lr_adjt: ', args.lr_adjt)
    print('==> backbone: ', args.backbone)
    print('==> use_pretrained: ', args.use_pretrained)

    # Hyper-parameters
    nb_epoch = args.epoch
    batch_size = args.batch
    num_workers = args.worker


    def get_transform(image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to target size
            transforms.RandomResizedCrop(image_size - 8, scale=(0.8, 1.0)),  # Crop slightly smaller
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Adjust if using pretrained ImageNet models
        ])


    def get_test_transform(image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size - 8),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    # Initial image size for training and testing
    image_size = 96
    transform_train = get_transform(image_size)
    transform_test = get_test_transform(image_size)

    # Data
    if args.dataset == 'Insect':
        data_dir = '/Projects/FAIR_Device_data/Zaki'
        train_list = '/sf_storage/Workspace/Zaki/HRN/insect_EffNet/insect_train_list_13Oct25.txt'
        test_list = '/sf_storage/Workspace/Zaki/HRN/insect_EffNet/insect_test_list_13Oct25.txt'
        trees = [
            [1],
            [1, 8],
            [1, 5, 15],
            [1, 11, 24],
            [1, 9, 25],
            [1, 10, 18, 39],
            [1, 5, 15, 34],
            [1, 7, 23, 40],
            [1, 7, 13, 35, 41],
            [1, 5, 15, 34, 42],
            [1, 7, 13, 26, 43],
            [1, 8, 17, 38, 46],
            [1, 5, 15, 28, 48],
            [1, 5, 15, 34, 51],
            [1, 5, 15, 34, 52],
            [1, 5, 15, 34, 53],
            [1, 5, 15, 34, 54],
            [1, 3, 22, 30, 55],
            [1, 5, 21, 31, 57],
            [1, 8, 17, 33, 58],
            [1, 5, 15, 28, 59],
            [1, 7, 13, 26, 61],
            [1, 3, 22, 30, 62],
            [1, 3, 22, 30, 44],
            [1, 3, 22, 30, 56],
            [1, 6, 19, 36, 45],
            [1, 12, 20, 37, 47],
            [1, 4, 16, 32, 49],
            [1, 5, 21, 29, 50],
            [0, 2, 14, 27, 60],
        ]
        # Removed 'levels = 5'. Length of path determined dynamically in tree_loss.py/generateStateSpace
        total_nodes = max(max(path) for path in trees) + 1
        print('total nodes: ', total_nodes)
        trainset = InsectDataset(data_dir, train_list, transform_train)
        # Uncomment this line for testing OA results
        testset = InsectDataset(data_dir, test_list, transform_test)
        # Uncomment this line for testing Average PRC results
        # testset = InsectDataset2(test_dir, transform_test, 'class', 1.0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              drop_last=True)  # Shuffle turned to 'False' for debugging
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             drop_last=True)

    # GPU
    device = torch.device("cuda:" + args.device[0])

    # Backbone setup
    backbone_name = args.backbone.lower()

    if backbone_name == 'resnet18':
        raw_backbone = models.resnet18(pretrained=False)
        num_ftrs = 512
        feature_size = 256
    elif backbone_name == 'resnet34':
        raw_backbone = models.resnet34(pretrained=False)
        num_ftrs = 512
        feature_size = 256
    elif backbone_name == 'resnet50':
        raw_backbone = models.resnet50(pretrained=False)
        num_ftrs = 2048
        feature_size = 1024
        # Load custom pretrained weights only for resnet50
        if args.use_pretrained:
            pretrained_path = '/sf_storage/Workspace/Zaki/HRN/pre-trained/resnet50-19c8e357.pth'
            raw_backbone.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    elif backbone_name == 'resnext101':
        raw_backbone = models.resnext101_32x8d(pretrained=False)
        num_ftrs = 2048
        feature_size = 1024
    elif backbone_name == 'efficientnetv2_s':
        raw_backbone = timm.create_model('efficientnetv2_s', pretrained=False)
        num_ftrs = 1280  # Output channels from EfficientNetV2-S
        feature_size = 640  # You can tune this based on your hierarchy depth
    elif backbone_name == 'mobilenetv3_small':
        raw_backbone = mobilenet_v3_small(pretrained=False)
        num_ftrs = 576  # Output channels from MobileNetV3 Small
        feature_size = 128
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if args.use_pretrained:
        checkpoint = torch.load(args.model, map_location='cpu')
        raw_backbone.load_state_dict(checkpoint)

    # Wrap the backbone to standardize output
    backbone = BackboneWrapper(raw_backbone)
    # print(backbone)

    # Model initialization
    net = HIFD2(backbone, feature_size=feature_size, num_ftrs=num_ftrs, dataset=args.dataset)
    net.to(device)

    # Loss functions
    # CELoss = nn.CrossEntropyLoss()
    tree = TreeLoss(trees, device, alpha=0.3, invert=True)

    optimizer = optim.SGD([
        {'params': net.classifier_pf4.parameters(), 'lr': 0.002},
        {'params': net.classifier_pf3.parameters(), 'lr': 0.002},
        {'params': net.classifier_pf2.parameters(), 'lr': 0.002},
        {'params': net.classifier_pf1.parameters(), 'lr': 0.002},
        {'params': net.classifier_leaf_sig.parameters(), 'lr': 0.002},
        {'params': net.classifier_leaf_soft.parameters(), 'lr': 0.002},
        {'params': net.fc_pf4.parameters(), 'lr': 0.002},
        {'params': net.fc_pf3.parameters(), 'lr': 0.002},
        {'params': net.fc_pf2.parameters(), 'lr': 0.002},
        {'params': net.fc_pf1.parameters(), 'lr': 0.002},
        {'params': net.fc_leaf_class.parameters(), 'lr': 0.002},
        {'params': net.conv_block_pf4.parameters(), 'lr': 0.002},
        {'params': net.conv_block_pf3.parameters(), 'lr': 0.002},
        {'params': net.conv_block_pf2.parameters(), 'lr': 0.002},
        {'params': net.conv_block_pf1.parameters(), 'lr': 0.002},
        {'params': net.conv_block_leaf_class.parameters(), 'lr': 0.002},
        {'params': net.backbone.parameters(), 'lr': 0.0002}
    ],
        momentum=0.9, weight_decay=5e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    # Get current date and time
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')

    save_name = f"{args.dataset}_{args.epoch}_{args.img_size}_bz{args.batch}_{args.backbone}_{args.lr_adjt}_{now}"
    train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt, args.dataset, tree,
          device, args.device, save_name, trainset, trees, get_transform, get_test_transform)

    # Evaluate OA
    test(net, testloader, tree, device, args.dataset, trainset,
         trees, get_test_transform, image_size)  # 'trainset' added to access label_to_name

    # Evaluate Average PRC
    # test_AP(net, args.dataset, testset, testloader, device)
