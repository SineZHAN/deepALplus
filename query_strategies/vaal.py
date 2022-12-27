import numpy as np
import torch
from .strategy import Strategy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad

'''
This implementation is with reference of https://github.com/sinhasam/vaal.
You need to write task-specific VAE in nets.py if you plan to apply this method in new task.
Please cite the original paper if you use this method.
@inproceedings{sinha2019variational,
  title={Variational adversarial active learning},
  author={Sinha, Samarth and Ebrahimi, Sayna and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5972--5981},
  year={2019}
}
'''
class VAAL(Strategy):
	def __init__(self, dataset, net, args_input, args_task, net_vae, net_dis, handler_joint):
		super(VAAL, self).__init__(dataset, net, args_input, args_task)
		self.net_vae = net_vae
		self.net_dis = net_dis
		self.handler_joint = handler_joint

	def query(self, n):
		unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
		self.train_vaal()
		uncertainties = self.pred_dis_score_vaal(unlabeled_data)
		return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

	def train_vaal(self, total_epoch=30,num_vae_steps=2, beta=1, adv_param=1):
		
		n_epoch = total_epoch
		num_vae_steps=num_vae_steps
		beta=beta
		adv_param=adv_param
		dim = self.dataset.X_train.shape[1:]
		self.vae = self.net_vae().cuda()
		self.dis = self.net_dis().cuda()
		if self.args_task['optimizer'] == 'Adam':
			opt_vae = optim.Adam(self.vae.parameters(), **self.args_task['optimizer_args'])
			opt_dis = optim.Adam(self.dis.parameters(), **self.args_task['optimizer_args'])
		elif self.args_task['optimizer'] == 'SGD':
			opt_vae = optim.SGD(self.vae.parameters(), **self.args_task['optimizer_args'])
			opt_dis = optim.SGD(self.dis.parameters(), **self.args_task['optimizer_args'])
		else:
			raise NotImplementedError

		#labeled and unlabeled data
		X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
		X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()


		loader_tr = DataLoader(self.handler_joint(X_labeled, Y_labeled,X_unlabeled, Y_unlabeled,
											transform = self.args_task['transform_train']), shuffle= True, **self.args_task['loader_tr_args'])
		
		for epoch in range(n_epoch):

			self.vae.train()
			self.dis.train()

			for index, label_x, label_y, unlabel_x, _ in loader_tr:

				label_x, label_y = label_x.cuda(), label_y.cuda()
				unlabel_x = unlabel_x.cuda()

				# vae
				for count in range(num_vae_steps):
					recon, z, mu, logvar = self.vae(label_x)
					unsup_loss = vae_loss(label_x, recon, mu, logvar, beta)
					unlabel_recon, unlabel_z, unlabel_mu, unlabel_logvar = self.vae(unlabel_x)
					transductive_loss = vae_loss(unlabel_x, unlabel_recon, unlabel_mu, unlabel_logvar, beta)
					label_preds = self.dis(mu)
					unlabel_preds = self.dis(unlabel_mu)

					label_preds_real = torch.ones(label_x.size(0)).cuda()
					unlabel_preds_real = torch.ones(unlabel_x.size(0)).cuda()
					bce_loss = nn.BCELoss()
					dsc_loss = bce_loss(label_preds, label_preds_real) + bce_loss(unlabel_preds, unlabel_preds_real)

					total_vae_loss = unsup_loss + transductive_loss + adv_param * dsc_loss

					opt_vae.zero_grad()
					total_vae_loss.backward()
					opt_vae.step()
				# disc
				for count in range(num_vae_steps):
					with torch.no_grad():
						_, _, mu, _ = self.vae(label_x)
						_, _, unlabel_mu, _ = self.vae(unlabel_x)
				
					label_preds = self.dis(mu)
					unlabel_preds = self.dis(unlabel_mu)
					
					label_preds_real = torch.ones(label_x.size(0)).cuda()
					unlabel_preds_real = torch.ones(unlabel_x.size(0)).cuda()
					
					bce_loss = nn.BCELoss()
					dsc_loss = bce_loss(label_preds, label_preds_real) + bce_loss(unlabel_preds, unlabel_preds_real)

					
					opt_dis.zero_grad()
					dsc_loss.backward()
					opt_dis.step()

	def pred_dis_score_vaal(self, data):
		loader_te = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])

		self.vae.eval()
		self.dis.eval()

		scores = torch.zeros(len(data))

		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.cuda(), y.cuda()
				_,_,mu,_ = self.vae(x)
				out = self.dis(mu).cpu()
				scores[idxs] = out.view(-1)

		return scores

def vae_loss(x, recon, mu, logvar, beta):
	mse_loss = nn.MSELoss()

	MSE = mse_loss(recon, x)
	
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	KLD = KLD * beta
	return MSE + KLD
