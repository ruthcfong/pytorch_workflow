import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms

import numpy as np

from sklearn.externals import joblib

import copy

from PIL import Image 

from architectures import LeNet, MnistNet, AlexNetCustom, alexnet_custom

#import matplotlib.pyplot as plt

import warnings

from collections import OrderedDict 

from custom import *

IMAGENET_MU = [0.485, 0.456, 0.406]
IMAGENET_SIGMA = [0.229, 0.224, 0.225]

CIFAR_MU = [0.4914, 0.4822, 0.4465]
CIFAR_SIGMA = [0.2023, 0.1994, 0.2010]

AIRCRAFT_MU = [0.4812, 0.5122, 0.5356]
AIRCRAFT_SIGMA = [0.2187, 0.2118, 0.2441]

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


"""
class NormalizedMSELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(NormalizedMSELoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        unnorm_loss = F.mse_loss(input, target, reduce=False)
        norm_loss = unnorm_loss / torch.abs(target + 1e-20)
        if not self.reduce:
            return norm_loss
        else:
            if self.size_average:
                return torch.mean(norm_loss)
            else:
                return torch.sum(norm_loss)
"""

class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        features = input.view(input.shape[0], input.shape[1], -1)
        gram_matrix = torch.bmm(features, features.transpose(1,2))
        return gram_matrix


class DiversityLoss(nn.Module):
    def __init__(self, size_average=True, use_gram=True, reduce=True):
        super(DiversityLoss, self).__init__()
        self.use_gram = use_gram
        if use_gram:
            self.gm = GramMatrix()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input):
        if self.use_gram:
            mats = [self.gm(x) for x in input] 
        else:
            mats = [x.view(x.shape[0], -1) for x in input]
        res = 0
        count = 0
        for a in range(len(input)):
            for b in range(len(input)):
                if a != b:
                    norm_a = torch.norm(mats[a].view(mats[a].shape[0], -1), dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(mats[a])
                    norm_b = torch.norm(mats[b].view(mats[b].shape[0], -1), dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(mats[b])
                    res = res + (mats[a] * mats[b]) / (norm_a * norm_b)
                    count += 1
        res = res / count
        if not self.reduce:
            return res
        if self.size_average:
            return torch.mean(res)
        else:
            return torch.sum(res)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, normalize=True):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin
        self.normalize=True

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        if self.normalize:
            euclidean_distance = 1 / float(np.prod(output1.shape[1])) * euclidean_distance
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


class NormalizedMSELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(NormalizedMSELoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        norm_input = torch.norm(input.view(input.shape[0], -1), p=2, dim=1)
        norm_target = torch.norm(target.view(target.shape[0], -1), p=2, dim=1)
        while len(norm_input.shape) < len(input.shape):
            norm_input.unsqueeze_(-1)
            norm_target.unsqueeze_(-1)
        norm_input = norm_input.expand_as(input)
        norm_target = norm_target.expand_as(target)
        loss = F.mse_loss(input / norm_input, target / norm_target, reduce=False)
        if not self.reduce:
            return loss 
        else:
            if self.size_average:
                return torch.mean(torch.sum(loss.view(loss.shape[0], -1)))
            else:
                return torch.sum(norm_loss)
        

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


def get_cifar10_class_name(label_i):
    return CIFAR10_CLASSES[label_i]


def get_short_imagenet_name(label_i, 
        label_names=np.loadtxt(os.path.join(BASE_REPO_PATH, 'synset_words.txt'), str, delimiter='\t')):
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


def get_input_size(dataset='imagenet', arch='alexnet'):
    if dataset == 'imagenet':
        if arch == 'alexnet':
            return [1,3,227,227]
        elif arch == 'inception_v3':
            return [1,3,299,299]
        else:
            return [1,3,224,224]
    elif dataset == 'mnist':
        return [1,1,28,28]
    elif dataset == 'cifar10' or dataset == 'cifar100':
        return [1,3,32,32]
    else:
        raise NotImplementedError
        assert(False)


def get_transform_detransform(dataset='imagenet', size=224, train=False):
    if dataset == 'mnist':
        return (transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]), 
                transforms.Compose([transforms.ToPILImage()]))
                #transforms.Compose([Clip(), transforms.ToPILImage()]))
    elif dataset == 'cifar10' or dataset == 'cifar100':
        assert(size == 32)
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MU, CIFAR_SIGMA)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MU, CIFAR_SIGMA)
            ])
        detransform = get_detransform(mu=CIFAR_MU, sigma=CIFAR_SIGMA)
        return (transform, detransform)
    elif dataset == 'imagenet':
        transform = get_transform(size=size, mu=IMAGENET_MU, 
                sigma=IMAGENET_SIGMA, train=train)
        detransform = get_detransform(mu=IMAGENET_MU, sigma=IMAGENET_SIGMA)
        return (transform, detransform)
    else:
        assert(False)


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


def get_model(arch, dataset='imagenet', adaptive_pool=False, pretrained=True, 
        checkpoint_path=None, cuda=False, **kwargs):
    """Returns a Pytorch model of the given architecture.

    TODO: Improve documentation for this function.

    Args:
        arch (str): Name of architecture (i.e., "alexnet", "vgg19", etc.).
        pretrained (bool): True if the returned model should use pretrained weights; False otherwise.
        cuda (bool): True if the returned model should be loaded onto available gpu(s); False otherwise.

    Returns:
        A Pytorch model.
    """
    if dataset == 'mnist' or dataset == 'svhn':
        in_channels = 1 if dataset == 'mnist' else 3
        if arch == 'lenet':
            if 'in_channels' in kwargs:
                in_channels = kwargs['in_channels']
            if 'activation' in kwargs:
                activation = kwargs['activation']
            else:
                activation = True
            if 'num_classes' in kwargs: 
                out_channels = kwargs['num_classes']
            else:
                out_channels = 10
            model = LeNet(in_channels=in_channels, out_channels=out_channels, 
                          activation=activation, adaptive_pool=adaptive_pool)
            model_path = os.path.join(BASE_REPO_PATH, 'models', 'lenet_model.pth.tar')
            assert(os.path.exists(model_path))
            if pretrained and checkpoint_path is None:
                assert(dataset == 'mnist')
                # load checkpoint originally trained using a GPU into the CPU
                # (see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3)
                checkpoint = torch.load(model_path, 
                        map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint['model'])
        elif arch == 'mnistnet':
            model = MnistNet(in_channels=in_channels)
            assert(not pretrained)
        elif arch == 'alexnet_custom':
            model = alexnet_custom(pretrained=pretrained, **kwargs)
        else:
            raise NotImplementedError
    elif dataset == 'cifar10' or dataset == 'cifar100':
        # requires pytorch-classification
        import models.cifar as cifar_models

        # architectures for which I have pretrained CIFAR-10/CIFAR-100 models
        CIFAR_ARCHS = ('alexnet_custom', 'lenet', 'alexnet', 'densenet', 'preresnet-110', 'resnet-110', 'vgg19_bn')

        if arch not in CIFAR_ARCHS:
            raise ValueError('Architecture "{}" for {} not found. Valid architectures are: {}'.format(
                             arch, dataset, ', '.join(CIFAR_ARCHS)))
        if arch == 'alexnet_custom':
            model = alexnet_custom(pretrained=pretrained, **kwargs)
        elif arch == 'lenet':
            model = LeNet(in_channels=3, adaptive_pool=adaptive_pool)
            assert(pretrained is False or checkpoint_path is not None)
        else:
            model = cifar_models.__dict__[arch](pretrained=pretrained, dataset=dataset) 
    #elif dataset == 'imagenet':
    else:
        if arch == 'alexnet_custom':
            model = alexnet_custom(pretrained=pretrained, **kwargs)
        else:
            model = models.__dict__[arch](pretrained=pretrained)
            if adaptive_pool:
                if arch == 'alexnet' or 'vgg' in arch:
                    module_name = 'features.%d' % (len(model.features)-1)
                    if arch == 'alexnet':
                        output_size = (6,6)
                    else:
                        output_size = (7,7)
                elif 'resnet' in arch:
                    s = model.fc.in_features / 512
                    assert(s == 1 or s == 4)
                    output_size = (s,s)
                else:
                    raise NotImplementedError
                model = replace_module(model, module_name.split('.'), nn.AdaptiveMaxPool2d(output_size))
    #else:
    #    raise NotImplementedError
    model.eval()
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path,
                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
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
    trunc_module, _ = truncate_module_helper(parent_module, module_path)
    return trunc_module


def truncate_module_helper(parent_module, module_path):
    if isinstance(parent_module, nn.Sequential):
        module_dict = OrderedDict()
    elif isinstance(parent_module, nn.Module):
        new_parent_module = copy.deepcopy(parent_module)
    seen_module = False
    for (k, v) in parent_module._modules.items():
        if seen_module and isinstance(parent_module, nn.Module):
            delattr(new_parent_module, v)
            continue

        if k == module_path[0]:
            if len(module_path) == 1:
                child_module = v 
                seen_module = True
            else:
                (child_module, seen_module) = truncate_module_helper(v, module_path[1:])
        else:
            child_module = v

        if isinstance(parent_module, nn.Sequential):
            module_dict[k] = child_module
        elif isinstance(parent_module, nn.Module):
            setattr(new_parent_module, k, child_module)
        else:
            assert(False)

        if seen_module and isinstance(parent_module, nn.Sequential):
            break

    if isinstance(parent_module, nn.Sequential):
        return (nn.Sequential(module_dict), seen_module)
    elif isinstance(parent_module, nn.Module):
        return (new_parent_module, seen_module)
    else:
        assert(False)


def get_first_module_name(module, running_name=''):
    if module._modules:
        next_module_name = list(module._modules)[0]
        if running_name == '':
            running_name = next_module_name
        else:
            running_name = running_name + '.' + next_module_name
        return get_first_module_name(module._modules[next_module_name],
                running_name=running_name)
    return running_name


def replace_max_with_avg_pool(parent_module):
    if isinstance(parent_module, nn.Sequential):
        module_dict = OrderedDict()
    elif isinstance(parent_module, nn.Module):
        new_parent_module = copy.deepcopy(parent_module)
    else:
        assert(False)
    for (k, v) in parent_module._modules.items():
        if isinstance(v, nn.MaxPool2d):
            assert(v.dilation == 1)
            child_module = nn.AvgPool2d(kernel_size=v.kernel_size, stride=v.stride, padding=v.padding, 
                                        ceil_mode=v.ceil_mode)
        elif len(v._modules.items()) > 0:
            child_module = replace_max_with_avg_pool(v)
        else:
            child_module = v
        
        if isinstance(parent_module, nn.Sequential):
            module_dict[k] = child_module
        elif isinstance(parent_module, nn.Module):
            setattr(new_parent_module, k, child_module)
    
    if isinstance(parent_module, nn.Sequential):
        return nn.Sequential(module_dict)
    elif isinstance(parent_module, nn.Module):
        return new_parent_module


activations = []

def hook_acts(module, input, output):
    activations.append(output)



def get_acts(model, input, second_input=None, clone=True):
    del activations[:]
    #if next(model.parameters()).is_cuda and not input.is_cuda:
    #    input = input.cuda()
    if second_input is not None:
        _ = model(input, second_input)
    else:
        _ = model(input)
    if clone:
        return [a.clone() for a in activations]
    else:
        return activations

gradients = []

def hook_grads(module, grad_input, grad_output):
    gradients.append(grad_input)


def get_grads(model, input, clone=True):
    del gradients[:]
    _ = model(input)
    if clone:
        return [g.clone() for g in gradients]
    else:
        return gradients


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


def hook_get_acts(model, blobs, input, second_input=None, features=None, quantile=None, threshold=None, clone=True):
    hooks = []
    for i in range(len(blobs)):
        hooks.append(get_pytorch_module(model, blobs[i]).register_forward_hook(hook_acts))

    acts_res = [a for a in get_acts(model, input, second_input=second_input, clone=clone)]
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


def hook_get_grads(model, blobs, input, clone=True):
    hooks = []
    for i in range(len(blobs)):
        hooks.append(get_pytorch_module(model, blobs[i]).register_backward_hook(hook_grads))

    grads_res = [a for a in get_acts(model, input, clone=clone)]

    for h in hooks:
        h.remove()

    return grads_res


def hook_get_shapes(model, blobs, input, features=None, clone=True):
    hooks = []
    for i in range(len(blobs)):
        hooks.append(get_pytorch_module(model, blobs[i]).register_forward_hook(hook_shapes))

    shapes_res = get_shapes(model, input, clone=clone)
    if features is not None:
        assert(len(blobs) == 1)
        shapes_res[0][1] = len(features)

    for h in hooks:
        h.remove()

    return shapes_res


class PCA(nn.Module):
    def __init__(self, pca_model, scaler_model=None):
        super(PCA, self).__init__()
        
        if scaler_model is not None:
            self.has_scale = True
            self.scale = nn.Parameter(torch.from_numpy(scaler_model.scale_).type(torch.FloatTensor))
            self.scale_mean = nn.Parameter(torch.from_numpy(scaler_model.mean_).type(torch.FloatTensor))
        else:
            self.has_scale = False
        
        if pca_model.mean_ is None:
            self.mean = nn.Parameter(torch.zeros(1))
        else:
            self.mean = nn.Parameter(torch.from_numpy(pca_model.mean_).type(torch.FloatTensor))
        self.components = nn.Parameter(torch.from_numpy(pca_model.components_).t().type(torch.FloatTensor))
        self.whiten = pca_model.whiten
        self.explained_variance = nn.Parameter(torch.from_numpy(pca_model.explained_variance_).type(torch.FloatTensor))

        self.n_components = pca_model.n_components_
        self.noise_variance = pca_model.noise_variance_
        self.singular_values = torch.from_numpy(pca_model.singular_values_).type(torch.FloatTensor)
        self.explained_variance_ratio = torch.from_numpy(pca_model.explained_variance_ratio_).type(torch.FloatTensor)
    
    def forward(self, x):
        if self.has_scale:
            x = x - self.scale_mean
            x = x / self.scale
        
        if self.mean is not None:
            x = x - self.mean
        
        x_transformed = torch.mm(x, self.components)
        
        if self.whiten:
            x_transformed = x_transformed / torch.sqrt(self.explained_variance)            
        
        return x_transformed


def load_pca_transform(pca_model_path, scaler_model_path=None):
    pca_model = joblib.load(pca_model_path) 
    if scaler_model_path is None:
        scaler_model = None
    else:
        scaler_model = joblib.load(scaler_model_path)
    return PCA(pca_model=pca_model, scaler_model=scaler_model)


def get_data_loader(dataset, **kwargs):
    if dataset == 'mnist':
        return get_mnist_data_loader(**kwargs)
    elif dataset == 'cifar10':
        return get_cifar10_data_loader(**kwargs)
    elif dataset == 'cifar100':
        return get_cifar100_data_loader(**kwargs)
    else:
        raise NotImplementedError


def get_cifar10_data_loader(datadir=CIFAR10_DATA_DIR, train=True, batch_size=64,
                            shuffle=False, cuda=False):
    transform, _ = get_transform_detransform(dataset='cifar10', size=32)
    dataset = datasets.CIFAR10(datadir, train=train, 
                               download=not os.path.exists(datadir),
                               transform=transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_cifar100_data_loader(datadir=CIFAR100_DATA_DIR, train=True, batch_size=64,
                            shuffle=False, cuda=False):
    transform, _ = get_transform_detransform(dataset='cifar100', size=32)
    dataset = datasets.CIFAR10(datadir, train=train, 
                               download=not os.path.exists(datadir),
                               transform=transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_mnist_data_loader(datadir=MNIST_DATA_DIR, train=True, batch_size=64, 
                          shuffle=False, normalize=False, cuda=False):
    if normalize:
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))
                                       ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(datadir, train=train,
        download=not os.path.exists(datadir), 
        transform=transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def save_checkpoint(state, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, filename)
    print('Saved model state dict at %s.' % filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).data.cpu().numpy()[0])
    return res


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


def show_image(img, title='', hide_ticks=True):
    f, ax = plt.subplots(1,1)
    ax.imshow(img)
    ax.set_title(title)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
