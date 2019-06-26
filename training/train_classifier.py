import argparse
import os
import random
import signal
import shutil
import socket
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

import transforms as custom_transforms

# Add visibility for subdirectories of the parent directory.
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.pardir)
sys.path.append(parent_dir)

import pytorch_utils as utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
dataset_names = ['imagenet', 'pascal']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-d', '--dataset', metavar='DATASET',
                    choices=dataset_names, default='imagenet',
                    help='dataset name: ' +
                         ' | '.join(dataset_names) +
                         ' (default: imagenet)')
parser.add_argument('--train_split', type=str, default='train',
                    help='name of training split')
parser.add_argument('--val_split', type=str, default='val',
                    help='name of validation split')
parser.add_argument('--year', choices=['2007', '2008', '2009', '2010',
                                       '2011', '2012'],
                    default='2012',
                    help='year of PASCAL dataset')
parser.add_argument('--download', action='store_true', default=False,
                    help='download dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-c', '--checkpoint', metavar='CHECKPOINT', type=str,
                    default='./checkpoint.pth.tar',
                    help='path to save checkpoint')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--finetune_last_layer', action='store_true', default=False,
                    help='if True, finetune the last layer of the model.')
parser.add_argument('--finetune_decay_factor', type=float, default=0.1,
                    help='factor by which to decay learning rate of '
                         'non-finetuned parameters.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--decay_factor', type=float, default=0.1,
                    help='factor by which to decay learning rate')
parser.add_argument('--decay_iter', type=int, default=30,
                    help='decay learning rate every X epochs')
parser.add_argument('--lr_schedule', type=str, choices=['step', 'multistep'],
                    default='multistep', help='type of lr scheduler')
parser.add_argument('--multistep_schedule', type=int, nargs='*', default=[30, 60, 80],
                    help='epochs at which to decrease learning rate')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_dir', default=None, type=str,
                    help='directory to save tensorboard logs to.')
parser.add_argument('-l', '--log_freq', default=1000, type=int,
                    metavar='N', help='log frequency (default: 1k).')
parser.add_argument('--checkpoint_freq', default=500, type=int,
                    help='checkpointing frequency (default: 500).')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

metrics = {}
metrics['lr'] = {}
metrics['train/acc1'] = {}
metrics['train/acc5'] = {}
metrics['train/loss'] = {}
metrics['val/acc1'] = {}
metrics['val/acc5'] = {}
metrics['val/loss'] = {}

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
def sig_handler(signum, frame):
    print("caught signal", signum)
    print(socket.gethostname(), "USR1 signal caught.")
    # do other stuff to cleanup here
    print('requeuing job ' + os.environ['SLURM_JOB_ID'])
    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)


def term_handler(signum, frame):
    print("bypassing sigterm", flush=True)


signal.signal(signal.SIGUSR1, sig_handler)
signal.signal(signal.SIGTERM, term_handler)
print('signal installed', flush=True)


def get_last_layer(model):
    """Returns a tuple containing the name and module of the last layer."""
    last_name, last_module = list(model.named_modules())[-1]
    return last_name, last_module


def get_model(arch, dataset="imagenet", pretrained=False):
    """Returns a model for the given dataset."""
    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()

    if dataset == "imagenet":
        num_classes = 1000
    elif dataset == "pascal":
        num_classes = 20
    else:
        assert False

    # Change the number of outputs in the last layer for non-ImageNet dataset.
    if dataset != "imagenet":
        # Get last layer.
        last_name, last_module = get_last_layer(model)

        # Construct new last layer.
        if isinstance(last_module, nn.Linear):
            in_features = last_module.in_features
            bias = last_module.bias is not None
            new_layer_module = nn.Linear(in_features, num_classes, bias=bias)
        else:
            assert(False)

        # Replace last layer.
        model = utils.replace_module(model,
                                     last_name.split('.'),
                                     new_layer_module)
    else:
        assert False

    return model

def get_criterion(dataset):
    """Return loss function based on dataset."""
    if dataset == "imagenet":
        criterion = nn.CrossEntropyLoss()
    elif dataset == "pascal":
        criterion = nn.BCEWithLogitsLoss()
    else:
        assert False
    return criterion


def main():
    args = parser.parse_args()

    # Set up tensorboard logging.
    if args.log_dir is not None:
        parent_dir = os.path.abspath(os.path.join(args.log_dir, os.pardir))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        logger = SummaryWriter(args.log_dir)
    else:
        logger = None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, logger, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, logger, args)


def main_worker(gpu, ngpus_per_node, logger, args):
    global best_acc1, metrics

    assert ".pth.tar" in args.checkpoint

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = get_model(args.arch,
                      dataset=args.dataset,
                      pretrained=args.pretrained)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = get_criterion(args.dataset).cuda(args.gpu)

    if args.finetune_last_layer:
        last_name, last_module = get_last_layer(model)
        optimizer = torch.optim.SGD(
            [
                {"params": [param for name, param in model.named_parameters()
                            if name != last_name],
                 "lr": args.lr * args.finetune_decay_factor,
                 },
                {"params": last_module.parameters()},
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        pass
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # resume checkpoint if available
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        start_iter = checkpoint['iter']
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        metrics = checkpoint['metrics']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.checkpoint, checkpoint['epoch']))
    else:
        start_iter = 0
        args.start_epoch = 0
        print("=> no checkpoint found at '{}'; creating a new one...".format(args.checkpoint))
        save_checkpoint({
            'epoch': args.start_epoch,
            'arch': args.arch,
            'iter': start_iter,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'metrics': metrics,
            'args': args,
        }, False, args.checkpoint)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == "imagenet":
        traindir = os.path.join(args.data, args.train_split)
        valdir = os.path.join(args.data, args.val_split)

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == "pascal":
        train_dataset = datasets.VOCDetection(
            args.data,
            year=args.year,
            image_set=args.train_split,
            download=args.download,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            target_transform=custom_transforms.VOCToClassVector()
        )

        val_dataset = datasets.VOCDetection(
            args.data,
            year=args.year,
            image_set=args.val_split,
            download=False,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            target_transform=custom_transforms.VOCToClassVector()
        )
    else:
        raise NotImplementedError("{} dataset not supported; only {}.".format(
            args.dataset, ' | '.join(dataset_names)
        ))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args.start_epoch, logger, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # save current learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        metrics['lr'][epoch] = curr_lr
        if logger is not None:
            logger.add_scalar('train/lr', curr_lr, epoch)

        # initialize start iteration
        start_iter_for_epoch = start_iter if epoch == args.start_epoch else 0

        # train for one epoch
        train_acc1, train_acc5, train_loss = train(train_loader,
                                                   model,
                                                   criterion,
                                                   optimizer,
                                                   epoch,
                                                   logger,
                                                   args,
                                                   start_iter=start_iter_for_epoch)

        # evaluate on validation set
        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, epoch, logger, args)

        metrics['train/acc1'][epoch] = train_acc1
        metrics['train/acc5'][epoch] = train_acc5
        metrics['train/loss'][epoch] = train_loss
        metrics['val/acc1'][epoch] = val_acc1
        metrics['val/acc5'][epoch] = val_acc5
        metrics['val/loss'][epoch] = val_loss

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'iter': 0,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'metrics': metrics,
                'args': args,
            }, is_best, args.checkpoint)


def train(train_loader, model, criterion, optimizer, epoch, logger, args, start_iter=0):
    global best_acc1, metrics

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # spoof sampler to continue from checkpoint without having to load
    # data unnecessarily
    _train_loader = train_loader.__iter__()
    if start_iter > 0:
        print(f"Moving index of iterator to last stopping point at {start_iter:d}/{len(_train_loader):d}.")
    for i in range(start_iter):
        try:
            next(_train_loader.sample_iter)
        except StopIteration:
            print(f"StopIteration hit at {i:d}/{len(_train_loader):d}.")
            break

    end = time.time()
    for i, (images, target) in enumerate(_train_loader):
        iter_num = i + start_iter

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_num % args.print_freq == 0:
            progress.print(iter_num)

        # log to tensorboard
        if logger is not None and iter_num % args.log_freq == 0:
            n_iter = epoch * len(train_loader) + iter_num
            logger.add_scalar('train/loss', loss.item(), n_iter)
            logger.add_scalar('train/acc1', acc1.item(), n_iter)
            logger.add_scalar('train/acc5', acc5.item(), n_iter)

        # save checkpoint
        if iter_num % args.checkpoint_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'iter': iter_num,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'metrics': metrics,
                'args': args,
            }, False, args.checkpoint)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, epoch, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if logger is not None:
            logger.add_scalar('val/loss', losses.avg, epoch)
            logger.add_scalar('val/acc1', top1.avg, epoch)
            logger.add_scalar('val/acc5', top5.avg, epoch)

    return top1.avg, top5.avg, losses.avg


def get_checkpoint_name(filename, suffix='_best'):
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    assert '.pth.tar' in filename
    index = basename.index('.pth.tar')
    basename_no_ext = basename[:index]
    ext = basename[index:]
    best_filename = os.path.join(directory, basename_no_ext + suffix + ext)
    return best_filename


def get_best_checkpoint_name(filename):
    return get_checkpoint_name(filename, suffix='_best')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Saved model state dict at {}'.format(filename))
    if is_best:
        best_filename = get_best_checkpoint_name(filename)
        shutil.copyfile(filename, best_filename)
        print('Save best model state dict at {}'.format(best_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate(s)."""
    if args.lr_schedule == 'multistep':
        exp = sum([1 if epoch >= milestone else 0
                   for milestone in args.multistep_schedule])
        lr = args.lr * (args.decay_factor ** exp)
    elif args.lr_schedule == 'step':
        lr = args.lr * (args.decay_factor ** (epoch // args.decay_iter))
    else:
        assert False
    if args.finetune_last_layer:
        assert len(optimizer.param_groups) == 2
        optimizer.param_groups[0]['lr'] = lr * args.finetune_decay_factor
        optimizer.param_groups[1]['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
