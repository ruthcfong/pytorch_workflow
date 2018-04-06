import os

import torch

import torch.nn as nn
from torch.autograd import Variable

from torchvision import transforms
from torchvision import models

import numpy as np
import copy

from PIL import Image 

import warnings

from collections import OrderedDict 

IMAGENET_MU = [0.485, 0.456, 0.406]
IMAGENET_SIGMA = [0.229, 0.224, 0.225]

BASE_PATH = "/home/ruthfong/pytorch-workflow"

class Clip(object):
    """Pytorch transformation that clips a tensor to be within [0,1]"""
    def __init__(self):
        return

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): tensor to be clipped.

        Returns:
            Tensor: clipped tensor.
        """
        t = tensor.clone()
        t[t > 1] = 1
        t[t < 0] = 0
        return t


def get_short_imagenet_name(label_i, 
        label_names=np.loadtxt(os.path.join(BASE_PATH, 'synset_words.txt'), str, delimiter='\t')):
    """Return the shortened name for an ImageNet index (zero-indexed).
    
    Args:
        label_i (int): index of ImageNet class (zero-index, [0, 999])
    
    Returns:
        str: short name of the given ImageNet class
    """
    return ' '.join(label_names[label_i-1].split(',')[0].split()[1:])


def set_gpu(gpu=None):
    """Set Pytorch to use the given gpu(s) (zero-indexed).

    Args:
        gpu (NoneType, int, or list of ints): the gpu(s) (zero-indexed) to use; None if no gpus should be used.

    Return:
        bool: True if using at least 1 gpu; otherwise False.
    """
    cuda = True if gpu is not None else False
    gpu_params = ''
    if cuda:
        if isinstance(gpu, list):
            gpu_params = str(gpu).strip('[').strip(']')
        else:
            gpu_params = str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_params
    print("%d GPU(s) being used at the following index(es): %s" % (torch.cuda.device_count(), gpu_params))
    return cuda


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_input_size(dataset='imagenet', network='alexnet'):
    if dataset == 'imagenet':
        if network == 'alexnet':
            return [1,3,227,227]
        elif network == 'inception_v3':
            return [1,3,299,299]
        else:
            return [1,3,224,224]
    elif dataset == 'mnist':
        return [1,1,28,28]
    else:
        assert(False)


def get_transform_detransform(dataset='imagenet'):
    if dataset == 'mnist':
        return (transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]), 
                transforms.Compose([transforms.ToPILImage()]))
                #transforms.Compose([Clip(), transforms.ToPILImage()]))
    elif dataset == 'imagenet':
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            #transforms.Scale(size=size),
            #transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma),
        ])

        detransform = transforms.Compose([
            Denormalize(mu, sigma),
            Clip(),
            transforms.ToPILImage(),
        ])

        return (transform, detransform)
    else:
        asesrt(False)


def get_transform(size=224, mu=IMAGENET_MU, sigma=IMAGENET_SIGMA, train=False):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma),
        ])
    return transform


def get_detransform(mu=IMAGENET_MU, sigma=IMAGENET_SIGMA):
    detransform = transforms.Compose([
        transforms.Normalize([-1*x for x in mu], [1./x for x in sigma]),
        Clip(),
        transforms.ToPILImage(),
    ])
    return detransform


def get_model(arch, pretrained=True, cuda=False):
    """Returns a Pytorch model of the given architecture.

    TODO: Improve documentation for this function.

    Args:
        arch (str): Name of architecture (i.e., "alexnet", "vgg19", etc.).
        pretrained (bool): True if the returned model should use pretrained weights; False otherwise.
        cuda (bool): True if the returned model should be loaded onto available gpu(s); False otherwise.

    Returns:
        A Pytorch model.
    """
    if arch == 'lenet':
        from architectures import LeNet
        model = LeNet()
        model_path = os.path.join(BASE_PATH, 'models', 'lenet_model.pth.tar')
        assert(os.path.exists(model_path))
        if pretrained:
            # load checkpoint originally trained using a GPU into the CPU
            # (see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3)
            checkpoint = torch.load(model_path, 
                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
    else:
        model = models.__dict__[arch](pretrained=pretrained)
    model.eval()
    if cuda:
        model.cuda()
    return model


def get_num_params(model):
    """Returns the number of parameters in a Pytorch model.

    Args:
        model: A Pytorch model.
    
    Return:
        int: number of parameters in the given Pytorch model.
    """
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum(np.prod(p.size()) for p in model_params)
    return num_params


def get_pytorch_module(net, blob):
    modules = blob.split('.')
    if len(modules) == 1:
        return net._modules.get(blob)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m


def get_all_blobs(parent_module, module_path=''):
    blob_accs = []
    for k, v in parent_module._modules.items():
        #print '.'.join([module_path, k])
        if module_path is '':
            blob_accs.extend(get_all_blobs(v, k))
        else:
            blob_accs.extend(get_all_blobs(v, '.'.join([module_path, k])))
    if module_path is not '':
        blob_accs.append(module_path)
    return blob_accs


def replace_module(parent_module, module_path, replacement_module):
    if isinstance(parent_module, nn.Sequential):
        module_dict = OrderedDict()
    elif isinstance(parent_module, nn.Module):
        new_parent_module = copy.deepcopy(parent_module)
    for (k, v) in parent_module._modules.items():
        if k == module_path[0]:
            if len(module_path) == 1:
                child_module = replacement_module
            else:
                child_module = replace_module(v, module_path[1:], replacement_module)
        else:
            child_module = v

        if isinstance(parent_module, nn.Sequential):
            module_dict[k] = child_module
        elif isinstance(parent_module, nn.Module):
            setattr(new_parent_module, k, child_module)
        else:
            assert(False)

    if isinstance(parent_module, nn.Sequential):
        return nn.Sequential(module_dict)
    elif isinstance(parent_module, nn.Module):
        return new_parent_module
    else:
        assert(False)


def truncate_module(parent_module, module_path):
    if isinstance(parent_module, nn.Sequential):
        module_dict = OrderedDict()
    elif isinstance(parent_module, nn.Module):
        new_parent_module = copy.deepcopy(parent_module)
    seen_module = False
    for (k, v) in parent_module._modules.items():
        if k == module_path[0]:
            if len(module_path) == 1:
                child_module = v 
                seen_module = True
            else:
                child_module = truncate_module(v, module_path[1:])
        else:
            child_module = v

        if isinstance(parent_module, nn.Sequential):
            module_dict[k] = child_module
        elif isinstance(parent_module, nn.Module):
            setattr(new_parent_module, k, child_module)
        else:
            assert(False)

        if seen_module:
            break

    if isinstance(parent_module, nn.Sequential):
        return nn.Sequential(module_dict)
    elif isinstance(parent_module, nn.Module):
        return new_parent_module
    else:
        assert(False)


activations = []

def hook_acts(module, input, output):
    activations.append(output)


def get_acts(model, input, clone=True):
    del activations[:]
    #if next(model.parameters()).is_cuda and not input.is_cuda:
    #    input = input.cuda()
    _ = model(input)
    if clone:
        return [a.clone() for a in activations]
    else:
        return activations


shapes = []

def hook_shapes(module, input, output):
    shapes.append(output.shape)


def get_shapes(model, input, clone=True):
    del shapes[:]
    _ = model(input)
    if clone:
        return [list(s) for s in shapes]
    else:
        return shapes


class Step(nn.Module):
    def __init__(self, threshold_value=0, min_value=0, max_value=1):
        super(Step, self).__init__()
        self.threshold_value = threshold_value
        self.min_value = min_value
        self.max_value = max_value
        self.min_threshold = nn.Threshold(self.threshold_value, self.min_value)
        self.max_threshold = nn.Threshold(-1*self.threshold_value, self.max_value)

    def forward(self, x):
       return self.max_threshold(-1*self.min_threshold(x))


def hook_get_acts(model, blobs, input, features=None, quantile=None, threshold=None, clone=True):
    hooks = []
    for i in range(len(blobs)):
        hooks.append(get_pytorch_module(model, blobs[i]).register_forward_hook(hook_acts))

    acts_res = [a for a in get_acts(model, input, clone=clone)]
    #acts_res = [a.detach() for a in get_acts(model, input, clone=clone)]
    if quantile is not None:
        quantile = torch.from_numpy(quantile).type(type(acts_res[0].data))
        quantile = Variable(quantile.cuda() if acts_res[0].is_cuda else quantile)
        acts_res = [(a > quantile.expand_as(a)).float() for a in acts_res]
        #print float(torch.sum(acts_res[0] == 1).float())
        #print torch.sum(acts_res[0] == 1).float() / float(np.prod(acts_res[0].shape))
    if features is not None:
        #if len(features) == 1:
        #    acts_res = [a[:,features].unsqueeze(1) for a in acts_res]
        #else:
        acts_res = [a[:,features] for a in acts_res]
    if threshold is not None:
        step = Step(threshold)
        if acts_res[0].is_cuda:
            step.cuda()
        acts_res = [step(a) for a in acts_res]
        #acts_res = [(a > threshold).type(a_data_type) for a in acts_res]
    #torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return acts_res


def hook_get_shapes(model, blobs, input, clone=True):
    hooks = []
    for i in range(len(blobs)):
        hooks.append(get_pytorch_module(model, blobs[i]).register_forward_hook(hook_shapes))

    shapes_res = get_shapes(model, input, clone=clone)

    for h in hooks:
        h.remove()

    return shapes_res

