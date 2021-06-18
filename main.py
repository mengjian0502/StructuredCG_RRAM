"""
model training
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
import os
import time
import models
import logging
from utils import *
from collections import OrderedDict

# reproducibility
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--lr_decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# Acceleration
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# channel gating
parser.add_argument('--slice_size', type=int, default=16, help='structured channel gating group size')
parser.add_argument('--cg_groups', type=int, default=1, help='apply channel gating if cg_groups > 1')
parser.add_argument('--cg_alpha', type=float, default=2.0, help='alpha value for channel gating')
parser.add_argument('--cg_threshold_init', type=float, default=0.0, help='initial threshold value for channel gating')
parser.add_argument('--cg_threshold_target', type=float, default=0.0, help='initial threshold value for channel gating')
parser.add_argument('--lambda_CG', type=float, default=0, help='lambda for Channel Gating regularization')
parser.add_argument('--share_by_kernel', type=int, default=0, help='block structure: [block_size, block_size, kernel_size, kernel_size]')
parser.add_argument("--hard_sig", type=str2bool, nargs='?', const=True, default=False, help="hardtanh gating function")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)
    
    # Preparing data
    if args.dataset == 'cifar10':
        data_path = args.data_path + 'cifar'

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 10
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val_')

        train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        num_classes = 1000
    else:
        raise ValueError("Dataset must be either cifar10 or imagenet_1k")  

    # Prepare the model
    logger.info('==> Building model..\n')
    model_cfg = getattr(models, args.model)

    if 'CG' in args.model:
        logger.info("cg_groups: {}".format(args.cg_groups))
        logger.info("cg_alpha: {0:.2f}".format(args.cg_alpha))
        logger.info("cg_threshold_init: {0:.2f}".format(args.cg_threshold_init))
        model_cfg.kwargs.update({"slice_size":args.slice_size, "cg_groups":args.cg_groups, "cg_threshold_init":args.cg_threshold_init, 
                                "cg_alpha":args.cg_alpha, "hard_sig":args.hard_sig})

    model_cfg.kwargs.update({"num_classes": num_classes})
    net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 
    logger.info(net)
    start_epoch = 0

    for name, value in net.named_parameters():
        logger.info(f'name: {name} | shape: {list(value.size())}')
    
    if args.fine_tune:
        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        for k, v in checkpoint['state_dict'].items():
            name = k
            new_state_dict[name] = v
        state_tmp = net.state_dict()

        if 'state_dict' in checkpoint.keys():
            state_tmp.update(new_state_dict)
            net.load_state_dict(state_tmp)
            logger.info("=> loaded checkpoint '{} | Acc={}'".format(args.resume, checkpoint['acc']))
        else:
            raise ValueError('no state_dict found')

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net)

    # Loss function
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=[lr_schedule])

    # Evaluate
    if args.evaluate:
        test_acc, val_loss = test(testloader, net, criterion)
        logger.info(f'Test accuracy: {test_acc}')
        exit()
    
    # Training
    start_time = time.time()
    epoch_time = AverageMeter()
    best_acc = 0.
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'best_acc']

    if args.cg_groups > 1:
        columns += ['cgspar', 'cgthre']

    for epoch in range(start_epoch, start_epoch+args.epochs):

        # Training phase
        train_results = train(trainloader, net, criterion, optimizer, args)
        # Test phase
        test_acc, val_loss = test(testloader, net, criterion)
        scheduler.step()

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        state = {
            'state_dict': net.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        filename='checkpoint.pth.tar'
        save_checkpoint(state, is_best, args.save_path, filename=filename)

        e_time = time.time() - start_time
        epoch_time.update(e_time)
        start_time = time.time()
        
        values = [epoch + 1, optimizer.param_groups[0]['lr'], train_results['loss'], train_results['acc'], e_time, val_loss, test_acc, best_acc]    
        if args.cg_groups > 1:
            _, cg_sparsity = get_cg_sparsity(net)
            cg_threshold = get_avg_cg_threshold(net)
            values += [cg_sparsity, cg_threshold]
    
        print_table(values, columns, epoch, logger)

def lr_schedule(epoch):
    t = epoch / args.epochs
    lr_ratio = 0.01
    
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return factor

def update_cg_threshold(model, epoch):
    if args.cg_threshold_target > 0:
        cg_thres = args.cg_threshold_target - (((args.epochs - 1 - epoch)/(args.epochs - 1)) ** 6.0) * (args.cg_threshold_target - args.cg_threshold_init)
        for m in model.modules():
            if hasattr(m, 'cg_threshold'):
                m.cg_threshold.data = torch.tensor(cg_thres)
    return cg_thres


if __name__ == '__main__':
    main()
