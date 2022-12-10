import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm

# LossPredictionLoss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already haved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

class Net_LPL:
    def __init__(self, net, params, device, net_lpl):
        self.net = net
        self.params = params
        self.device = device
        self.net_lpl = net_lpl
        
    def train(self, data, weight = 1.0, margin = 1.0 , lpl_epoch = 20):
        n_epoch = self.params['n_epoch']
        n_epoch = lpl_epoch + self.params['n_epoch']
        epoch_loss = lpl_epoch

        dim = data.X.shape[1:]
        self.clf = self.net(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
        self.clf_lpl = self.net_lpl.to(self.device)
        #self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError
        optimizer_lpl = optim.Adam(self.clf_lpl.parameters(), lr = 0.01)

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        self.clf.train()
        self.clf_lpl.train()
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                optimizer_lpl.zero_grad()
                out, feature = self.clf(x)
                out, e1 = self.clf(x)
                cross_ent = nn.CrossEntropyLoss(reduction='none')
                target_loss = cross_ent(out,y)
                if epoch >= epoch_loss:
                    feature[0] = feature[0].detach()
                    feature[1] = feature[1].detach()
                    feature[2] = feature[2].detach()
                    feature[3] = feature[3].detach()
                pred_loss = self.clf_lpl(feature)
                pred_loss = pred_loss.view(pred_loss.size(0))

                backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                module_loss = LossPredLoss(pred_loss, target_loss, margin)
                loss = backbone_loss + weight * module_loss
                loss.backward()
                optimizer.step()
                optimizer_lpl.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
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
    
    def get_grad_embeddings(self, data):
        self.clf.eval()
        embDim = self.clf.get_embedding_dim()
        nLab = self.params['num_class']
        embeddings = np.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) * -1.0
                        else:
                            embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) * -1.0

        return embeddings
        
class MNIST_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.conv = nn.Conv2d(1, 3, kernel_size = 1)
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	def forward(self, x):
		x = self.conv(x)
		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	
	def get_embedding_dim(self):
		return self.dim

class CIFAR10_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(512, num_classes)

		self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.dim = resnet18.fc.in_features

		
	
	def forward(self, x):

		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim

class openml_Net(nn.Module):
    def __init__(self, dim = 28 * 28, embSize=256, pretrained=False, num_classes = 10):
        super(openml_Net, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, num_classes)
    
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    
    def get_embedding_dim(self):
        return self.embSize

class PneumoniaMNIST_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(512, num_classes)

		self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.feature0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.dim = resnet18.fc.in_features
	
	
	def forward(self, x):

		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim

class waterbirds_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(resnet18.fc.in_features, num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim


def get_lossnet(name):
	if name == 'PneumoniaMNIST':
		return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'MNIST' in name:
		return LossNet(feature_sizes=[14, 7, 4, 2], num_channels=[64, 128, 256, 512], interm_dim=128) 
	elif 'CIFAR' in name:
		return LossNet(feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'ImageNet' in name:
		return LossNet(feature_sizes=[64, 32, 16, 8], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'BreakHis' in name:
		return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'waterbirds' in name:
		return LossNet(feature_sizes=[128, 64, 32, 16], num_channels=[64, 128, 256, 512], interm_dim=128)
	else:
		raise NotImplementedError

class LossNet(nn.Module):
	def __init__(self, feature_sizes=[28, 14, 7, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
		super(LossNet, self).__init__()
		
		self.GAP1 = nn.AvgPool2d(feature_sizes[0])
		self.GAP2 = nn.AvgPool2d(feature_sizes[1])
		self.GAP3 = nn.AvgPool2d(feature_sizes[2])
		self.GAP4 = nn.AvgPool2d(feature_sizes[3])

		self.FC1 = nn.Linear(num_channels[0], interm_dim)
		self.FC2 = nn.Linear(num_channels[1], interm_dim)
		self.FC3 = nn.Linear(num_channels[2], interm_dim)
		self.FC4 = nn.Linear(num_channels[3], interm_dim)

		self.linear = nn.Linear(4 * interm_dim, 1)
	
	def forward(self, features):

		out1 = self.GAP1(features[0])
		out1 = out1.view(out1.size(0), -1)
		out1 = F.relu(self.FC1(out1))

		out2 = self.GAP2(features[1])
		out2 = out2.view(out2.size(0), -1)
		out2 = F.relu(self.FC2(out2))

		out3 = self.GAP3(features[2])
		out3 = out3.view(out3.size(0), -1)
		out3 = F.relu(self.FC3(out3))

		out4 = self.GAP4(features[3])
		out4 = out4.view(out4.size(0), -1)
		out4 = F.relu(self.FC4(out4))

		out = self.linear(torch.cat((out1, out2, out3, out4), 1))
		return out
