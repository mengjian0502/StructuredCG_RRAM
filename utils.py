"""
Utilities of training
"""
from models.cg import Conv2d_CG
import os
import time
import shutil
import tabulate
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from pruner import glasso_global_mp, get_weight_sparsity
from models.cg import QConv2d_CG

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

class Hook_record_input():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input

    def close(self):
        self.hook.remove()

def add_input_record_Hook(model, name_as_key=False):
    Hooks = {}
    if name_as_key:
        for name,module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                Hooks[name] = Hook_record_input(module)
            
    else:
        for k,module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                Hooks[k] = Hook_record_input(module)
    return Hooks


def get_cg_sparsity(model):
    total = 0.
    nonzeros = 0.
    total_g = 0.
    nonzero_g = 0.

    for m in model.modules():
        if hasattr(m, 'num_out'):
            total += m.num_out
            nonzeros += m.num_full

            total_g += m.total_g
            nonzero_g += m.nonzero_g

    cg_sparsity = (total - nonzeros) / total
    cg_grp_sparsity = (total_g - nonzero_g) / total_g
    return cg_sparsity, cg_grp_sparsity

def get_avg_cg_threshold(model):
    cg_threshold_list = []
    for m in model.modules():
        if hasattr(m, 'cg_threshold'):
            cg_threshold_list.append(m.cg_threshold.data.view(-1))
    cg_threshold_all = torch.cat(cg_threshold_list)
    return cg_threshold_all.mean().cpu().numpy()

def get_avg_cg_threshold(model):
    cg_threshold_list = []
    for m in model.modules():
        if hasattr(m, 'cg_threshold'):
            cg_threshold_list.append(m.cg_threshold.data.view(-1))
    cg_threshold_all = torch.cat(cg_threshold_list)
    return cg_threshold_all.mean().cpu().numpy()

def getChannelGatingRegularizer(model, target_cg_threshold):
    """
    regularzier for the learnable cg threshold
    """
    Loss = 0
    for name, param in model.named_parameters():
        if 'cg_threshold' in name:
            Loss += torch.sum((param-target_cg_threshold)**2)
    return Loss

def train(trainloader, net, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure the data loading time
        data_time.update(time.time() - end)
        
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()
    
        outputs = net(inputs)
        loss = criterion(outputs, targets) 

        if args.swp:
            # compute the global threshold
            thre1 = net.get_global_mp_thre(args.ratio)

            lamda = torch.tensor(args.lamda).cuda()
            reg_g1 = torch.tensor(0.).cuda()
            thre_list = []
            penalty_groups = 0
            count = 0
            for m in net.modules():
                if isinstance(m, Conv2d_CG):
                    if not count in [0]:
                        w_l = m.weight
                        kw = m.weight.size(2)
                        if kw != 1:
                            w_l = w_l.permute(0,2,3,1)
                            w_l = w_l.contiguous().view(w_l.size(0)*w_l.size(1)*w_l.size(2), w_l.size(3))
                            
                            reg1, penalty_group1 = glasso_global_mp(w_l, dim=1, thre=thre1)
                            reg_g1 += reg1
                            thre = thre1
                            penalty_group = penalty_group1
                        
                        # pruning statistics
                        if batch_idx == len(trainloader) - 1:
                            thre_list.append(thre)                      
                            penalty_groups += penalty_group
                    count += 1

            loss += lamda * (reg_g1)
        else:
            penalty_groups = 0
            thre_list = []

        if args.lambda_CG > 0:
            loss += args.lambda_CG * getChannelGatingRegularizer(net, args.cg_threshold_target)

        reg_alpha = torch.tensor(0.).cuda()
        a_lambda = torch.tensor(args.weight_decay).cuda()

        alpha = []
        for name, param in net.named_parameters():
            if 'alpha' in name:
                alpha.append(param.item())
                reg_alpha += param.item() ** 2
        loss += a_lambda * (reg_alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
         
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    
    res = {
        'acc':top1.avg,
        'loss':losses.avg
    } 
    return res


def test(testloader, net, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    test_loss = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            test_loss += loss.item()

            batch_time.update(time.time() - end)
            end = time.time()
    return top1.avg, losses.avg

def update_optimizer(optimizer):
    for p in optimizer.param_groups:
        param = p['params'][0]
        import pdb;pdb.set_trace()
        if len(param.size()) == 4 and param.size(1) > 3:
            lr = p['lr']

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def adjust_learning_rate_schedule(optimizer, epoch, gammas, schedule, lr, mu):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')

def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 

if __name__ == "__main__":
    log = log2df('./save/strucCG/resnet20_CG/resnet20_CG_optimSGD_lr0.1_wd0.0005_cg2_cg_slideFalse_4bit_swp/resnet20_CG_optimSGD_lr0.1_wd0.0005_swp0.0003.log')
    epoch = log['ep']
    train_acc = log['tr_acc']
    test_acc = log['te_acc']
    grp_spar = log['grp_spar']
    ovall_spar = log['cgspar']
    spar_groups = log['spar_groups']
    cg_spars = log['cgspar']

    table = {
        'epoch': epoch,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'grp_spar': grp_spar,
        'cg_spar': cg_spars,
        'ovall_spar': ovall_spar,
        'spar_groups':spar_groups,
    }

    variable = pd.DataFrame(table, columns=['epoch', 'train_acc', 'test_acc', 'grp_spar','cg_spars', 'ovall_spar', 'spar_groups'])
    variable.to_csv('resnet20_CG_optimSGD_lr0.1_wd0.0005_cg2_swp0.0003.csv', index=False)
