import argparse
import numpy as np
import warnings
import torch
from utils import get_dataset, get_net, get_net_lpl, get_net_waal, get_strategy
from pprint import pprint

torch.set_printoptions(profile='full')

import sys
import os
import re
import random
import math
import datetime

import arguments
from parameters import *
from utils import *

# parameters
args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy


SEED = args_input.seed
os.environ['TORCH_HOME']='./basicmodel'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled  = True
torch.backends.cudnn.benchmark= True

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#recording
sys.stdout = Logger(os.path.abspath('') + '/logfile/' + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_log.txt')
warnings.filterwarnings('ignore')

# start experiment

iteration = args_input.iteration

all_acc = []
acq_time = []

# repeate # iteration trials
while (iteration > 0): 
	iteration = iteration - 1

	# data, network, strategy
	args_task = args_pool[DATA_NAME]
	dataset = get_dataset(args_input.dataset_name, args_task)				# load dataset
	if args_input.ALstrategy == 'LossPredictionLoss':
		net = get_net_lpl(args_input.dataset_name, args_task, device)		# load network
	elif args_input.ALstrategy == 'WAAL':
		net = get_net_waal(args_input.dataset_name, args_task, device)		# load network
	else:
		net = get_net(args_input.dataset_name, args_task, device)			# load network
	strategy = get_strategy(args_input.ALstrategy, dataset, net, args_input, args_task)  # load strategy

	start = datetime.datetime.now()


	# generate initial labeled pool
	dataset.initialize_labels(args_input.initseed)

	#record acc performance
	acc = np.zeros(NUM_ROUND + 1)

	# only for special cases that need additional data
	new_X = torch.empty(0)
	new_Y = torch.empty(0)
		
	# print info
	print(DATA_NAME)
	print('RANDOM SEED {}'.format(SEED))
	print(type(strategy).__name__)
	
	# round 0 accuracy
	if args_input.ALstrategy == 'WAAL':
		strategy.train(model_name = args_input.ALstrategy)
	else:
		strategy.train()
	preds = strategy.predict(dataset.get_test_data())
	acc[0] = dataset.cal_test_acc(preds)
	print('Round 0\ntesting accuracy {}'.format(acc[0]))
	print('\n')
	
	# round 1 to rd
	for rd in range(1, NUM_ROUND+1):
		print('Round {}'.format(rd))
		high_confident_idx = []
		high_confident_pseudo_label = []
		# query
		if 'CEALSampling' in args_input.ALstrategy:
			q_idxs, new_data = strategy.query(NUM_QUERY, rd, option = args_input.ALstrategy[13:])
		else:
			q_idxs = strategy.query(NUM_QUERY)
	
		# update
		strategy.update(q_idxs)

		#train
		if 'CEALSampling' in args_input.ALstrategy:
			strategy.train(new_data)
		elif args_input.ALstrategy == 'WAAL':
			strategy.train(model_name = args_input.ALstrategy)
		else:
			strategy.train()
	
		# round rd accuracy
		preds = strategy.predict(dataset.get_test_data())
		acc[rd] = dataset.cal_test_acc(preds)
		print('testing accuracy {}'.format(acc[rd]))
		print('\n')

		#torch.cuda.empty_cache()
	
	# print results
	print('SEED {}'.format(SEED))
	print(type(strategy).__name__)
	print(acc)
	all_acc.append(acc)
	
	#save model
	timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	model_path = './modelpara/'+timestamp + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota)  +'.params'
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds),3))
	torch.save(strategy.get_model().state_dict(), model_path)
	
# cal mean & standard deviation
acc_m = []
file_name_res_tot = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res_tot.txt'
file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res_tot),'w')

file_res_tot.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res_tot.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res_tot.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res_tot.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
file_res_tot.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
file_res_tot.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res_tot.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res_tot.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')

# result
for i in range(len(all_acc)):
	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	print(str(i)+': '+str(acc_m[i]))
	file_res_tot.writelines(str(i)+': '+str(acc_m[i])+'\n')
mean_acc,stddev_acc = get_mean_stddev(acc_m)
mean_time, stddev_time = get_mean_stddev(acq_time)

print('mean AUBC(acc): '+str(mean_acc)+'. std dev AUBC(acc): '+str(stddev_acc))
print('mean time: '+str(mean_time)+'. std dev time: '+str(stddev_time))

file_res_tot.writelines('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc)+'\n')
file_res_tot.writelines('mean time: '+str(mean_time)+'. std dev acc: '+str(stddev_time)+'\n')

# save result

file_name_res = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res.txt'
file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')


file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
file_res.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')
avg_acc = np.mean(np.array(all_acc),axis=0)
for i in range(len(avg_acc)):
	tmp = 'Size of training set is ' + str(NUM_INIT_LB + i*NUM_QUERY) + ', ' + 'accuracy is ' + str(round(avg_acc[i],4)) + '.' + '\n'
	file_res.writelines(tmp)

file_res.close()
file_res_tot.close()
