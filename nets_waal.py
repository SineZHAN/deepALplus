import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable, grad
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init

class Net_WAAL:
	def __init__(self, net, params, device, net_fea, net_clf, net_dis, handler_joint):
		self.net = net
		self.params = params
		self.device = device
		self.net_fea = net_fea
		self.net_clf = net_clf
		self.net_dis = net_dis
		self.handler_joint = handler_joint
		
	def train(self, data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled, alpha = 1e-3):
		n_epoch = self.params['n_epoch']

		dim = data.X.shape[1:]
		self.fea = self.net_fea(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
		self.clf = self.net_clf(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
		self.dis = self.net_dis(dim = self.fea.get_embedding_dim()).to(self.device)

		# setting three optimizers
		if self.params['optimizer'] == 'Adam':
			opt_fea = optim.Adam(self.fea.parameters(), **self.params['optimizer_args'])
			opt_clf = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
			opt_dis = optim.Adam(self.dis.parameters(), **self.params['optimizer_args'])
		elif self.params['optimizer'] == 'SGD':
			opt_fea = optim.SGD(self.fea.parameters(), **self.params['optimizer_args'])
			opt_clf = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
			opt_dis = optim.SGD(self.dis.parameters(), **self.params['optimizer_args'])
		else:
			raise NotImplementedError
		

		# computing the unbalancing ratio, a value betwwen [0,1]
		#gamma_ratio = X_labeled.shape[0]/X_unlabeled.shape[0]
		gamma_ratio = 1
		loader_tr = DataLoader(self.handler_joint(X_labeled, Y_labeled,X_unlabeled, Y_unlabeled,
											transform = self.params['transform_train']), shuffle= True, **self.params['loader_tr_args'])
		for epoch in tqdm(range(1, n_epoch+1), ncols=100):
			# setting the training mode in the beginning of EACH epoch
			# (since we need to compute the training accuracy during the epoch, optional)
			self.fea.train()
			self.clf.train()
			self.dis.train()

			for index, label_x, label_y, unlabel_x, _ in loader_tr:

				label_x, label_y = label_x.to(self.device), label_y.to(self.device)
				unlabel_x = unlabel_x.to(self.device)

				# training feature extractor and predictor
				self.set_requires_grad(self.fea,requires_grad=True)
				self.set_requires_grad(self.clf,requires_grad=True)
				self.set_requires_grad(self.dis,requires_grad=False)

				#print(label_x.shape)
				lb_z   = self.fea(label_x)
				unlb_z = self.fea(unlabel_x)

				opt_fea.zero_grad()
				opt_clf.zero_grad()

				lb_out, _ = self.clf(lb_z)

				# prediction loss (deafult we use F.cross_entropy)
				pred_loss = torch.mean(F.cross_entropy(lb_out,label_y))

				# Wasserstein loss (unbalanced loss, used the redundant trick)
				wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

				with torch.no_grad():

					lb_z = self.fea(label_x)
					unlb_z = self.fea(unlabel_x)

				gp = self.gradient_penalty(self.dis, unlb_z, lb_z)

				loss = pred_loss + alpha * wassertein_distance + alpha * gp * 5
				# for CIFAR10 the gradient penality is 5
				# for SVHN the gradient penality is 2

				loss.backward()
				opt_fea.step()
				opt_clf.step()


				# Then the second step, training discriminator

				self.set_requires_grad(self.fea, requires_grad=False)
				self.set_requires_grad(self.clf, requires_grad=False)
				self.set_requires_grad(self.dis, requires_grad=True)


				with torch.no_grad():

					lb_z = self.fea(label_x)
					unlb_z = self.fea(unlabel_x)


				for _ in range(1):

					# gradient ascent for multiple times like GANS training

					gp = self.gradient_penalty(self.dis, unlb_z, lb_z)

					wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

					dis_loss = -1 * alpha * wassertein_distance - alpha * gp * 2

					opt_dis.zero_grad()
					dis_loss.backward()
					opt_dis.step()


	def predict(self, data):
		self.clf.eval()
		self.fea.eval()
		preds = torch.zeros(len(data), dtype=data.Y.dtype)
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				latent  = self.fea(x)
				out, _  = self.clf(latent)
				pred	= out.max(1)[1]
				preds[idxs] = pred.cpu()
		return preds
	
	def predict_prob(self, data):
		self.clf.eval()
		probs = torch.zeros([len(data), len(np.unique(data.Y))])
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				latent = self.fea(x)
				out, _ = self.clf(latent)
				prob = F.softmax(out, dim=1)
				probs[idxs] = prob.cpu()
		return probs
	
	def single_worst(self, probas):

		"""
		The single worst will return the max_{k} -log(proba[k]) for each sample
		:param probas:
		:return:  # unlabeled \times 1 (tensor float)
		"""

		value,_ = torch.max(-1*torch.log(probas),1)

		return value

	# setting gradient values
	def set_requires_grad(self, model, requires_grad=True):
		"""
		Used in training adversarial approach
		:param model:
		:param requires_grad:
		:return:
		"""

		for param in model.parameters():
			param.requires_grad = requires_grad


	# setting gradient penalty for sure the lipschitiz property
	def gradient_penalty(self, critic, h_s, h_t):
		''' Gradeitnt penalty approach'''
		alpha = torch.rand(h_s.size(0), 1).to(self.device)
		differences = h_t - h_s
		interpolates = h_s + (alpha * differences)
		interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
		# interpolates.requires_grad_()
		preds = critic(interpolates)
		gradients = grad(preds, interpolates,
						 grad_outputs=torch.ones_like(preds),
						 retain_graph=True, create_graph=True)[0]
		gradient_norm = gradients.norm(2, dim=1)
		gradient_penalty = ((gradient_norm - 1)**2).mean()

		return gradient_penalty 

	
	def get_model(self):
		return self.clf

	def get_embeddings(self, data):
		self.clf.eval()
		embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf(x)
				embeddings[idxs] = e1.cpu()
		return embeddings
	
		
class MNIST_Net_WAAL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.conv = nn.Conv2d(1, 3, kernel_size = 1)
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		x = self.conv(x)

		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)
		return x
	
	def get_embedding_dim(self):
		return self.dim

class CIFAR10_Net_WAAL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.classifier = nn.Linear(512,num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		feature  = self.features(x)
		#print(x)s
		x = feature.view(feature.size(0), -1)
		return x
	
	def get_embedding_dim(self):
		return self.dim

class waterbirds_Net_WAAL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		feature  = self.features(x)
		#print(x)
		x = feature.view(feature.size(0), -1)
		return x
	
	def get_embedding_dim(self):
		return self.dim

class CLF_WAAL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):	
		output = self.classifier(x)
		return output, x
	
	def get_embedding_dim(self):
		return self.dim

class Discriminator(nn.Module):
		"""Adversary architecture(Discriminator) for WAE-GAN."""
		def __init__(self, dim=32):
				super(Discriminator, self).__init__()
				self.dim = np.prod(dim)
				self.net = nn.Sequential(
						nn.Linear(self.dim, 512),
						nn.ReLU(True),
						nn.Linear(512, 512),
						nn.ReLU(True),
						nn.Linear(512,1),
						nn.Sigmoid(),
				)
				self.weight_init()

		def weight_init(self):
				for block in self._modules:
						for m in self._modules[block]:
								kaiming_init(m)

		def forward(self, z):
				return self.net(z).reshape(-1)

def kaiming_init(m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				init.kaiming_normal(m.weight)
				if m.bias is not None:
						m.bias.data.fill_(0)
		elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
				m.weight.data.fill_(1)
				if m.bias is not None:
						m.bias.data.fill_(0)

def normal_init(m, mean, std):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				m.weight.data.normal_(mean, std)
				if m.bias.data is not None:
						m.bias.data.zero_()
		elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				m.weight.data.fill_(1)
				if m.bias.data is not None:
						m.bias.data.zero_()
