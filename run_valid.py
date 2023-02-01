import os

import cv2
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='0'


from kaggle_rsna_v2 import *
from sklearn import metrics

from common  import *
from my_lib.net.lookahead import *
from model import *
from dataset import *
from sklearn import metrics

import torch.cuda.amp as amp
is_amp = True  #True #False

#################################################################################################


def np_binary_cross_entropy_loss(probability, truth):
	probability = probability.astype(np.float64)
	probability = np.nan_to_num(probability, nan=1, posinf=1, neginf=0)

	p = np.clip(probability,1e-5,1-1e-5)
	y = truth

	loss = -y * np.log(p) - (1-y)*np.log(1-p)
	loss = loss.mean()
	return loss

def get_f1score(probability, truth):
    f1score = []
    threshold = np.linspace(0, 1, 50)
    for t in threshold:
        predict = (probability > t).astype(np.float32)

        tp = ((predict>=0.5) & (truth>=0.5)).sum()
        fp = ((predict>=0.5) & (truth< 0.5)).sum()
        fn = ((predict< 0.5) & (truth>=0.5)).sum()

        r = tp/(tp+fn+1e-3)
        p = tp/(tp+fp+1e-3)
        f1 = 2*r*p/(r+p+1e-3)
        f1score.append(f1)
    f1score=np.array(f1score)
    return f1score, threshold




#######################################################################

fold = 0
out_dir  = root_dir + '/result/run300/kaggle/effb0-1536-baseline-aug00-00'
fold_dir = out_dir  + f'/fold-{fold}'
initial_checkpoint =\
	None #fold_dir + '/checkpoint/00000000.model.pth'  #None #root_dir + '/result/run62b/coat-full-1024-flip1b-global-extern-vindr/fold-0/checkpoint/00005520.model.pth'  #
	#

def run_valid():
	global fold_dir

	log = Logger()
	log.open(fold_dir+'/log.submit.txt',mode='a')


	train_df, valid_df = read_kaggle_csv()
	valid_dataset = RsnaDataset(valid_df)
	valid_loader = DataLoader(
		valid_dataset,
		sampler = SequentialSampler(valid_dataset),
		batch_size  = 1,
		drop_last   = False,
		num_workers = 8,
		pin_memory  = False,
		collate_fn = null_collate,
	)
	log.write(f'valid_dataset : \n{valid_dataset}\n')

	#-----
	net = Net().cuda()
	f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	net.load_state_dict(state_dict, strict=True)  # True

	#-----

	valid_num = 0
	valid = dotdict(
		cancer = dotdict(
			truth=[],
			predict=[],
		),
		# birads = dotdict(
		# 	truth=[],
		# 	predict=[],
		# ),
		# density = dotdict(
		# 	truth=[],
		# 	predict=[],
		# ),
	)

	net = net.eval()
	start_timer = timer()
	for t, batch in enumerate(valid_loader):

		batch_size = len(batch['index'])
		for k in tensor_key: batch[k] = batch[k].cuda()

		net.output_type = ['loss', 'inference']
		with torch.no_grad():
			with amp.autocast(enabled = is_amp):
				output = net(batch)#data_parallel(net, batch) #
				#loss0  = output['vindr_abnormality_loss'].mean()

		valid_num += batch_size
		valid.cancer.truth.append(batch['cancer'].data.cpu().numpy())
		valid.cancer.predict.append(output['cancer'].data.cpu().numpy())
		#show_result(batch, output, wait=0)

		#---
		print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
		#if valid_num==200*4: break

	assert(valid_num == len(valid_loader.dataset))
	cancer_t   = np.concatenate(valid.cancer.truth)
	cancer_p   = np.concatenate(valid.cancer.predict)

	valid_df.loc[:,'cancer_p'] = cancer_p
	if 1:
		valid_loader.dataset.df.to_csv(f'{fold_dir}/valid/valid_df.csv',index=False)
		np.save(f'{fold_dir}/valid/cancer_t.npy',cancer_t)
		np.save(f'{fold_dir}/valid/cancer_p.npy',cancer_p)

	#------
	cancer_loss  = np_binary_cross_entropy_loss(cancer_p, cancer_t)

	fpr, tpr, thresholds = metrics.roc_curve(cancer_t, cancer_p)
	auc = metrics.auc(fpr, tpr)

	f1score, threshold = get_f1score(cancer_p, cancer_t)
	i = f1score.argmax()
	f1score, threshold = f1score[i], threshold[i]

	specificity = ((cancer_p<threshold) &((cancer_t<=0.5))).sum() / (cancer_t<=0.5).sum()
	sensitivity = ((cancer_p>=threshold)&((cancer_t>=0.5))).sum() / (cancer_t>=0.5).sum()


	#---
	gb = valid_df[['patient_id', 'laterality', 'cancer', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
	f1score_mean, threshold_mean = get_f1score(gb.cancer_p, gb.cancer)
	i = f1score_mean.argmax()
	f1score_mean, threshold_mean = f1score_mean[i], threshold_mean[i]

	print(
		[ cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]
	)

def plot_auc(cancer_p, cancer_t):
	cancer_t = cancer_t.astype(int)
	pos, bin = np.histogram(cancer_p[cancer_t == 1], np.linspace(0, 1, 20))
	neg, bin = np.histogram(cancer_p[cancer_t == 0], np.linspace(0, 1, 20))
	pos = pos / (cancer_t == 1).sum()
	neg = neg / (cancer_t == 0).sum()
	print(pos)
	print(neg)
	# plt.plot(bin[1:],neg, alpha=1)
	# plt.plot(bin[1:],pos, alpha=1)
	bin = (bin[1:] + bin[:-1]) / 2
	plt.bar(bin, neg, width=0.05, label='neg',alpha=0.5)
	plt.bar(bin, pos, width=0.05, label='pos',alpha=0.5)
	plt.legend()
	plt.show()

def run_more():
	valid_df = pd.read_csv(f'{fold_dir}/valid/valid_df.csv')

	cancer_t  = np.load(f'{fold_dir}/valid/cancer_t.npy', )
	cancer_p  = np.load(f'{fold_dir}/valid/cancer_p.npy', )

	valid_df.loc[:,'cancer_p'] = cancer_p
	valid_df.loc[:,'cancer_t'] = valid_df.cancer
	valid_df = valid_df[valid_df.site_id==1].reset_index(drop=True)

	#---

	f1score, threshold = get_f1score(valid_df.cancer_p, valid_df.cancer_t)
	i = f1score.argmax()
	f1score, threshold = f1score[i], threshold[i]

	fpr, tpr, thresholds = metrics.roc_curve(valid_df.cancer_t, valid_df.cancer_p)
	auc = metrics.auc(fpr, tpr)


	print('single image')
	print('f1score', f1score)
	print('threshold', threshold)
	print('auc', auc)
	print('')

	#---
	gb = valid_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
	f1score_mean, threshold_mean = get_f1score(gb.cancer_p, gb.cancer_t)
	i = f1score_mean.argmax()
	f1score_mean, threshold_mean = f1score_mean[i], threshold_mean[i]

	fpr, tpr, thresholds = metrics.roc_curve(gb.cancer_t, gb.cancer_p)
	auc_mean = metrics.auc(fpr, tpr)


	print('groupby mean')
	print('f1score_mean', f1score_mean)
	print('threshold_mean', threshold_mean)
	print('auc_mean', auc_mean)
	print('')

	plot_auc(gb.cancer_p, gb.cancer_t)

	# ---
	gb = valid_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).max()
	f1score_max, threshold_max = get_f1score(gb.cancer_p, gb.cancer_t)
	i = f1score_max.argmax()
	f1score_max, threshold_max = f1score_max[i], threshold_max[i]

	fpr, tpr, thresholds = metrics.roc_curve(gb.cancer_t, gb.cancer_p)
	auc_max = metrics.auc(fpr, tpr)

	print('groupby max')
	print('f1score_max', f1score_max)
	print('threshold_max', threshold_max)
	print('auc_mean', auc_max)
	print('')



# main #################################################################
if __name__ == '__main__':
	#run_valid()
	run_more()

'''
 

'''
