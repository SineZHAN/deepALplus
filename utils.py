from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler, openml_Handler, MNIST_Handler_joint, SVHN_Handler_joint, CIFAR10_Handler_joint
from data import get_MNIST, get_FashionMNIST, get_EMNIST, get_SVHN, get_CIFAR10, get_CIFAR10_imb, get_CIFAR100,  \
								get_TinyImageNet, get_openml, get_BreakHis, get_PneumoniaMNIST, get_waterbirds
from nets import Net, MNIST_Net, CIFAR10_Net, openml_Net, PneumoniaMNIST_Net, waterbirds_Net, get_net_vae
from nets_lossprediction import Net_LPL, MNIST_Net_LPL, CIFAR10_Net_LPL, PneumoniaMNIST_Net_LPL, waterbirds_Net_LPL, get_lossnet
from nets_waal import Net_WAAL, MNIST_Net_WAAL, CIFAR10_Net_WAAL, waterbirds_Net_WAAL, CLF_WAAL, Discriminator
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
								LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
								KMeansSampling, KMeansSamplingGPU, KCenterGreedy, KCenterGreedyPCA, BALDDropout,  \
								AdversarialBIM, AdversarialDeepFool, VarRatio, MeanSTD, BadgeSampling, CEALSampling, \
								LossPredictionLoss, VAAL, WAAL
from parameters import *
from torchvision import transforms
import sys
import os
import numpy as np
import math
import torch


#Handler

def get_handler(name):
	if name == 'MNIST':
		return MNIST_Handler
	elif name == 'MNIST_pretrain':
		return MNIST_Handler
	elif name == 'FashionMNIST':
		return MNIST_Handler
	elif name == 'EMNIST':
		return MNIST_Handler
	elif name == 'SVHN':
		return SVHN_Handler
	elif name == 'CIFAR10':
		return CIFAR10_Handler
	elif name == 'CIFAR10_imb':
		return CIFAR10_Handler
	elif name == 'CIFAR100':
		return CIFAR10_Handler
	elif name == 'TinyImageNet':
		return CIFAR10_Handler
	elif name == 'openml':
		return openml_Handler
	elif name == 'BreakHis':
		return CIFAR10_Handler
	elif name == 'PneumoniaMNIST':
		return CIFAR10_Handler
	elif name == 'waterbirds':
		return CIFAR10_Handler
	elif name == 'waterbirds_pretrain':
		return CIFAR10_Handler
	else: 
		raise NotImplementedError

def get_handler_joint(name):
	if name == 'MNIST':
		return MNIST_Handler_joint
	elif name == 'MNIST_pretrain':
		return MNIST_Handler_joint
	elif name == 'FashionMNIST':
		return MNIST_Handler_joint
	elif name == 'EMNIST':
		return MNIST_Handler_joint
	elif name == 'SVHN':
		return SVHN_Handler_joint
	elif name == 'CIFAR10':
		return CIFAR10_Handler_joint
	elif name == 'CIFAR10_imb':
		return CIFAR10_Handler_joint
	elif name == 'CIFAR100':
		return CIFAR10_Handler_joint
	elif name == 'TinyImageNet':
		return CIFAR10_Handler_joint
	elif name == 'openml':
		raise NotImplementedError
	elif name == 'BreakHis':
		return CIFAR10_Handler_joint
	elif name == 'PneumoniaMNIST':
		return CIFAR10_Handler_joint
	elif name == 'waterbirds':
		return CIFAR10_Handler_joint
	elif name == 'waterbirds_pretrain':
		return CIFAR10_Handler_joint
	else: 
		raise NotImplementedError

def get_dataset(name, args_task):
	if name == 'MNIST':
		return get_MNIST(get_handler(name), args_task)
	elif name == 'MNIST_pretrain':
		return get_MNIST(get_handler(name), args_task)
	elif name == 'FashionMNIST':
		return get_FashionMNIST(get_handler(name), args_task)
	elif name == 'EMNIST':
		return get_EMNIST(get_handler(name), args_task)
	elif name == 'SVHN':
		return get_SVHN(get_handler(name), args_task)
	elif name == 'CIFAR10':
		return get_CIFAR10(get_handler(name), args_task)
	elif name == 'CIFAR10_imb':
		return get_CIFAR10_imb(get_handler(name), args_task)
	elif name == 'CIFAR100':
		return get_CIFAR100(get_handler(name), args_task)
	elif name == 'TinyImageNet':
		return get_TinyImageNet(get_handler(name), args_task)
	elif name == 'openml':
		return get_openml(get_handler(name), args_task)
	elif name == 'BreakHis':
		return get_BreakHis(get_handler(name), args_task)
	elif name == 'PneumoniaMNIST':
		return get_PneumoniaMNIST(get_handler(name), args_task)
	elif name == 'waterbirds':
		return get_waterbirds(get_handler(name), args_task)
	elif name == 'waterbirds_pretrain':
		return get_waterbirds(get_handler(name), args_task)
	else:
		raise NotImplementedError

#net

def get_net(name, args_task, device):
	if name == 'MNIST':
		return Net(MNIST_Net, args_task, device)
	elif name == 'MNIST_pretrain':
		return Net(MNIST_Net, args_task, device)
	elif name == 'FashionMNIST':
		return Net(MNIST_Net, args_task, device)
	elif name == 'EMNIST':
		return Net(MNIST_Net, args_task, device)
	elif name == 'SVHN':
		return Net(CIFAR10_Net, args_task, device)
	elif name == 'CIFAR10':
		return Net(CIFAR10_Net, args_task, device)
	elif name == 'CIFAR10_imb':
		return Net(CIFAR10_Net, args_task, device)
	elif name == 'CIFAR100':
		return Net(CIFAR10_Net, args_task, device)
	elif name == 'TinyImageNet':
		return Net(CIFAR10_Net, args_task, device)
	elif name == 'openml':
		return Net(openml_Net, args_task, device)
	elif name == 'BreakHis':
		return Net(CIFAR10_Net, args_task, device)
	elif name == 'PneumoniaMNIST':
		return Net(PneumoniaMNIST_Net, args_task, device)
	elif name == 'waterbirds':
		return Net(waterbirds_Net, args_task, device)
	elif name == 'waterbirds_pretrain':
		return Net(waterbirds_Net, args_task, device)
	else:
		raise NotImplementedError

def get_net_lpl(name, args_task, device):
	loss_net = get_lossnet(args_task['name'])
	if name == 'MNIST':
		return Net_LPL(MNIST_Net_LPL, args_task, device, loss_net)
	elif name == 'MNIST_pretrain':
		return Net_LPL(MNIST_Net_LPL, args_task, device, loss_net)
	elif name == 'FashionMNIST':
		return Net_LPL(MNIST_Net_LPL, args_task, device, loss_net)
	elif name == 'EMNIST':
		return Net_LPL(MNIST_Net_LPL, args_task, device, loss_net)
	elif name == 'SVHN':
		return Net_LPL(CIFAR10_Net_LPL, args_task, device, loss_net)
	elif name == 'CIFAR10':
		return Net_LPL(CIFAR10_Net_LPL, args_task, device, loss_net)
	elif name == 'CIFAR10_imb':
		return Net_LPL(CIFAR10_Net_LPL, args_task, device, loss_net)
	elif name == 'CIFAR100':
		return Net_LPL(CIFAR10_Net_LPL, args_task, device, loss_net)
	elif name == 'TinyImageNet':
		return Net_LPL(CIFAR10_Net_LPL, args_task, device, loss_net)
	elif name == 'openml':
		raise NotImplementedError
	elif name == 'BreakHis':
		return Net_LPL(CIFAR10_Net_LPL, args_task, device, loss_net)
	elif name == 'PneumoniaMNIST':
		return Net_LPL(PneumoniaMNIST_Net_LPL, args_task, device, loss_net)
	elif name == 'waterbirds':
		return Net_LPL(waterbirds_Net_LPL, args_task, device, loss_net)
	elif name == 'waterbirds_pretrain':
		return Net_LPL(waterbirds_Net_LPL, args_task, device, loss_net)
	else:
		raise NotImplementedError

def get_net_waal(name, args_task, device):
	handler_joint = get_handler_joint(args_task['name'])
	# note that the first function parameter (net) in Net_WAAL is useless
	if name == 'MNIST':
		return Net_WAAL(MNIST_Net_WAAL, args_task, device, MNIST_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'MNIST_pretrain':
		return Net_WAAL(MNIST_Net_WAAL, args_task, device, MNIST_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'FashionMNIST':
		return Net_WAAL(MNIST_Net_WAAL, args_task, device, MNIST_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'EMNIST':
		return Net_WAAL(MNIST_Net_WAAL, args_task, device, MNIST_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'SVHN':
		return Net_WAAL(MNIST_Net_WAAL, args_task, device, MNIST_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'CIFAR10':
		return Net_WAAL(CIFAR10_Net_WAAL, args_task, device, CIFAR10_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'CIFAR10_imb':
		return Net_WAAL(CIFAR10_Net_WAAL, args_task, device, CIFAR10_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'CIFAR100':
		return Net_WAAL(CIFAR10_Net_WAAL, args_task, device, CIFAR10_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'TinyImageNet':
		return Net_WAAL(CIFAR10_Net_WAAL, args_task, device, CIFAR10_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'openml':
		raise NotImplementedError
	elif name == 'BreakHis':
		return Net_WAAL(CIFAR10_Net_WAAL, args_task, device, CIFAR10_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'PneumoniaMNIST':
		return Net_WAAL(MNIST_Net_WAAL, args_task, device, MNIST_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'waterbirds':
		return Net_WAAL(waterbirds_Net_WAAL, args_task, device, waterbirds_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	elif name == 'waterbirds_pretrain':
		return Net_WAAL(waterbirds_Net_WAAL, args_task, device, waterbirds_Net_WAAL, CLF_WAAL, Discriminator, handler_joint)
	else:
		raise NotImplementedError	
#params

def get_params(name):
	return args_pool[name]

#strategy

def get_strategy(STRATEGY_NAME, dataset, net, args_input, args_task):
	if STRATEGY_NAME == 'RandomSampling':
		return RandomSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'LeastConfidence':
		return LeastConfidence(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'MarginSampling':
		return MarginSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'EntropySampling':
		return EntropySampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'LeastConfidenceDropout':
		return LeastConfidenceDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'MarginSamplingDropout':
		return MarginSamplingDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'EntropySamplingDropout':
		return EntropySamplingDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'KMeansSampling':
		return KMeansSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'KMeansSamplingGPU':
		return KMeansSamplingGPU(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'KCenterGreedy':
		return KCenterGreedy(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'KCenterGreedyPCA':
		return KCenterGreedyPCA(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'BALDDropout':
		return BALDDropout(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'VarRatio':
		return VarRatio(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'MeanSTD':
		return MeanSTD(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'BadgeSampling':
		return BadgeSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'LossPredictionLoss':		
		return LossPredictionLoss(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'AdversarialBIM':
		return AdversarialBIM(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'AdversarialDeepFool':
		return AdversarialDeepFool(dataset, net, args_input, args_task)
	elif 'CEALSampling' in STRATEGY_NAME:
		return CEALSampling(dataset, net, args_input, args_task)
	elif STRATEGY_NAME == 'VAAL':	
		net_vae,net_disc = get_net_vae(args_task['name'])
		handler_joint = get_handler_joint(args_task['name'])
		return VAAL(dataset, net, args_input, args_task, net_vae = net_vae, net_dis = net_disc, handler_joint = handler_joint)
	elif STRATEGY_NAME == 'WAAL':
		return WAAL(dataset, net, args_input, args_task)
	else:
		raise NotImplementedError




#other stuffs

# logger
class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)

def get_aubc(quota, bsize, resseq):
	# it is equal to use np.trapz for calculation
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum
