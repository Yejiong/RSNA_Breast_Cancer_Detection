import os
import pdb
import dotdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'


# from kaggle_rsna_v2 import *
from sklearn import metrics

# from common  import *
# from my_lib.net.lookahead import *
from model import *
from dataset import *

import torch.cuda.amp as amp
is_amp = True  #True #False


meta = {
	'mass': [0, 0, 255],
	'calc': [255, 255, 0],
	'dist': [0, 255, 0],
	'asym': [0, 255, 255],
	'other': [128, 128, 128],
}
def draw_abnormality_heatmap(image, abnormality_heatmap, contrast=0.5,interpolation=cv2.INTER_LINEAR,):

	color = np.array(list(meta.values())) / 255
	name = list(meta.keys())

	overlay = cv2.applyColorMap(((image ** 0.75) * 255).astype(np.uint8), cv2.COLORMAP_BONE) / 255
	h, w, _ = overlay.shape

	overlay = overlay * (contrast)
	for i, a in enumerate(abnormality_heatmap):
		a = cv2.resize(a, dsize=(w, h), interpolation=interpolation)

		# overlay[...,0] = overlay[...,0] + a*0.5
		overlay = overlay + a.reshape(w, h, 1) * (1-contrast) * [[color[i]]]

	overlay = np.clip(overlay, 0, 1)
	return overlay


def show_cam_heatmap(batch, output, net, wait=0):
	batch_size =len(batch['index'])
	image    = batch['image'].squeeze(1).float().data.cpu().numpy()
	cancer_t = batch['cancer'].float().data.cpu().numpy()
	heatmap  = output['cam'].float().data.cpu().numpy()
	cancer   = output['cancer'].float().data.cpu().numpy()


	for b in range(batch_size):
		h = heatmap[b]
		m = image[b]
		q = cancer[b]
		height, width = m.shape

		h = (h - h.min()) / (h.max() - h.min() + 1e-8)
		h = cv2.resize(h, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

		overlay = cv2.addWeighted(
			cv2.cvtColor((m* 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
			0.5,
			cv2.applyColorMap((h * q * 255).astype(np.uint8), cv2.COLORMAP_JET),
			0.5, 0
		)
		# draw_shadow_text(overlay, f'study_id {study_id}', (5, 50), 1.25, (255, 255, 255), 2)
		# draw_shadow_text(overlay, f'image_id {image_id}', (5, 100), 1.25, (255, 255, 255), 2)
		# draw_shadow_text(overlay, f'{q:0.6f}', (5, 150), 1.55, (255, 255, 255), 3)

		print(cancer_t[b],q)
		image_show('m', m, resize=1)
		image_show('overlay', overlay, resize=1)
		cv2.waitKey(wait)


def show_result(batch, output, wait=0):
	batch_size =len(batch['index'])

	image = batch['image'].squeeze(2).float().data.cpu().numpy()
	heatmap_t = batch['vindr_abnormality_heatmap'].float().data.cpu().numpy()
	abnormality_t = batch['vindr_abnormality'].float().data.cpu().numpy()
	density_t = batch['vindr_density'].float().data.cpu().numpy()
	birads_t = batch['vindr_birads'].float().data.cpu().numpy()

	heatmap_p = output['vindr_abnormality_heatmap'].float().data.cpu().numpy()
	abnormality_p = output['vindr_abnormality'].float().data.cpu().numpy()
	density_p = output['vindr_density'].float().data.cpu().numpy()
	birads_p = output['vindr_birads'].float().data.cpu().numpy()

	color = np.array(list(meta.values())) / 255
	name = list(meta.keys())

	if image_height<=512:
		font_size, font_thickness = 0.75, 1
		font_space = 20
		resize=1
	if image_height>512:
		font_size, font_thickness = 1.25, 2
		font_space = 34
		resize=0.5

	for b in range(batch_size):
		study_id = batch['df'][b]['study_id'].values[0]

		m = image[b]
		h_t = heatmap_t[b]
		h_p = heatmap_p[b]


		#h_p = h_p/(0.0001+h_p.reshape(5,-1).max(-1).reshape(5,1,1))
		#h_p[0] = h_p[0]/(0.0001+h_p[0].reshape(5,-1).max(-1).reshape(5,1,1))*abnormality_p[b, 0,].reshape(5,1,1)
		#h_p[1] = h_p[1]/(0.0001+h_p[1].reshape(5,-1).max(-1).reshape(5,1,1))*abnormality_p[b, 1,].reshape(5,1,1)

		h_p = np.concatenate([h_p[0], np.flip(h_p[1],axis=2)], 2)
		h_t = np.concatenate([h_t[0], np.flip(h_t[1],axis=2)], 2)
		m = np.hstack([m[0], np.flip(m[1],axis=1)])

		overlay_t = draw_abnormality_heatmap(m, h_t, contrast=0.5, interpolation=cv2.INTER_NEAREST)
		overlay_p = draw_abnormality_heatmap(m, h_p, contrast=0.5, interpolation=cv2.INTER_LINEAR)
		for i in range(5):
			draw_shadow_text(overlay_p,
							 f'{abnormality_p[b, 0, i]:.3f}|{abnormality_t[b, 0, i]:.0f}  {abnormality_p[b, 1, i]:.3f}|{abnormality_t[b, 1, i]:.0f}  {name[i]}',
							 (5, (i + 1) * font_space), font_size, color[i], font_thickness)

		draw_shadow_text(overlay_p,
						 f'{density_p[b, 0].argmax()}|{density_t[b, 0]:.0f}  {density_p[b, 1].argmax()}|{density_t[b, 0]:.0f}  density',
						 (5, image_height - font_space), font_size, (1, 1, 1), font_thickness)
		draw_shadow_text(overlay_p,
						 f'{birads_p[b, 0].argmax()}|{birads_t[b, 0]:.0f}  {birads_p[b, 1].argmax()}|{birads_t[b, 1]:.0f}  birads',
						 (5, image_height - 2*font_space), font_size, (1, 1, 1), font_thickness)
		draw_shadow_text(overlay_p, study_id, (5, image_height - 3*font_space), font_size, (0.5, 0.5, 0.5), font_thickness)

		image_show_norm('image', m, min=0, max=1, resize=resize)
		# image_show_norm('overlay_t', overlay_t, min=0, max=1, resize=resize)
		# image_show_norm('overlay_p', overlay_p, min=0, max=1, resize=resize)
		image_show_norm('overlay_p', np.hstack([overlay_t,overlay_p]), min=0, max=1, resize=resize)
		cv2.waitKey(wait)

#################################################################################################
def np_softmax(x, axis = -1):
	x = x.astype(np.float64)
	x = np.nan_to_num(x, nan=50, posinf=50, neginf=-50)
	x = x - np.max(x, axis=axis, keepdims=True)
	x_exp = np.exp(x)
	return x_exp / x_exp.sum(axis=axis, keepdims=True)

def np_onehot(y, num_class):
	one_hot = np.eye(num_class)[y]
	return one_hot

def np_cross_entropy_loss(probability, truth):
	probability = probability.astype(np.float64)
	probability = np.nan_to_num(probability, nan=1, posinf=1, neginf=0)
	truth = truth.astype(np.int32)

	num, num_class = probability.shape
	p = np.clip(probability,1e-5,1-1e-5)
	y = np_onehot(truth, num_class)

	loss = -np.sum(y * np.log(p))/num
	return loss


def np_f1_score(predict, truth, num_class):
	f1score=[]
	for c in range(num_class):
		tp = ((predict==c) & (truth==c)).sum()
		fp = ((predict==c) & (truth!=c)).sum()
		fn = ((predict!=c) & (truth==c)).sum()

		precision = tp/(tp+fp+0.0001)
		recall = tp/(tp+fn+0.0001)
		s = 2*precision*recall/(precision+recall+0.0001)
		f1score.append(s)
	return f1score


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



def do_valid(net, valid_loader):

	valid_num = 0
	
	# valid = dotdict(
	# 	cancer = dotdict(
	# 		truth=[],
	# 		predict=[],
	# 	),
	# 	# birads = dotdict(
	# 	# 	truth=[],
	# 	# 	predict=[],
	# 	# ),
	# 	# density = dotdict(
	# 	# 	truth=[],
	# 	# 	predict=[],
	# 	# ),
	# )

	valid = {"truth": [], "predict": [],}
 
	# pdb.set_trace()
 
	net = net.eval()
	start_timer = time.time()
	for t, batch in enumerate(valid_loader):

		batch_size = len(batch['index'])
		for k in tensor_key: batch[k] = batch[k].cuda()

		net.output_type = ['loss', 'inference']
		with torch.no_grad():
			with amp.autocast(enabled = is_amp):
				# pdb.set_trace()
				output = net(batch)
				# output = data_parallel(net, batch) #net(input)#
				#loss0  = output['vindr_abnormality_loss'].mean()


		valid_num += batch_size
		# valid.cancer.truth.append(batch['cancer'].data.cpu().numpy())
		# valid.cancer.predict.append(output['cancer'].data.cpu().numpy())

		valid["truth"].append(batch['cancer'].data.cpu().numpy())
		valid["predict"].append(output['cancer'].data.cpu().numpy())

		#show_result(batch, output, wait=0)

		#---
		print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),(time.time() - start_timer,'sec')),end='',flush=True)
		#if valid_num==200*4: break
  
	assert(valid_num == len(valid_loader.dataset))
	cancer_t   = np.concatenate(valid["truth"])
	cancer_p   = np.concatenate(valid["predict"])


	valid_df = valid_loader.dataset.df.copy()
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

	return [ cancer_loss, auc, f1score, threshold, sensitivity, specificity, f1score_mean]


 ##----------------
fold = 0
# out_dir  = root_dir + '/result/run300/kaggle/effb0-1536-baseline-aug00-00'
out_dir = "./tmp"
# fold_dir = out_dir  + f'/fold-{fold}'
fold_dir = out_dir  + f'valid_df.fold{fold}.csv'
initial_checkpoint =\
	None #fold_dir + '/checkpoint/00000000.model.pth'  #None #root_dir + '/result/run62b/coat-full-1024-flip1b-global-extern-vindr/fold-0/checkpoint/00005520.model.pth'  #
	#

def run_train():

	start_lr   = 1e-4 #0.0001
	batch_size = 8 #32 #32
	skip_save_epoch = 3
	num_epoch = 10000000000000
	ratio = 8


	debug_mode = 'view-none' #'view-none'#'view-valid'#'view-train'


	# def scheduler(epoch):
	#
	# 	if epoch<1:
	# 		lr = 1e-3
	# 	elif epoch<4:
	# 		lr = 1e-4
	# 	else:
	# 		lr = 1e-5
	#
	# 	return lr

	def scheduler(epoch):
		#return start_lr

		num_epoch=6
		start_lr = 1e-3
		min_lr = 1e-5
		#min_lr = 3e-4

		lr = (num_epoch-epoch)/num_epoch * (start_lr-min_lr) + min_lr
		lr = max(min_lr,lr)
		return lr

	if 0:
		epoch = np.linspace(0,6,100)
		lr = [scheduler(e) for e in epoch]
		plt.plot(epoch, lr)
		plt.yscale('log')
		plt.show()
		exit(0)

	## setup  ----------------------------------------
	for f in ['checkpoint','train','valid','backup'] : os.makedirs(fold_dir +'/'+f, exist_ok=True)
	#backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

	# log = Logger()
	# log.open(fold_dir+'/log.train.txt',mode='a')
	# log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}\n\n')
	# log.write(f'\t{set_environment()}\n')
	# log.write(f'\t__file__ = {__file__}\n')
	# log.write(f'\tfold_dir = {fold_dir}\n' )
	# log.write('\n')

	
	## dataset ----------------------------------------
	# log.write('** dataset setting **\n')
	print('** dataset setting **\n')
	#train_df, valid_df = make_fold(fold)
	train_df, valid_df = read_kaggle_csv()
	
	#train_df = train_df[train_df.cancer==1].reset_index(drop=True)
	train_dataset = RsnaDataset(train_df, augment=train_augment_v00)
	valid_dataset = RsnaDataset(valid_df)
	#valid_df.to_csv(f'valid_df.fold{fold}.csv',index=False)
	#exit(0)
	
	
	train_loader  = DataLoader(
		train_dataset,
		#sampler = RandomSampler(train_dataset),
		# sampler = BalanceSampler(train_dataset, ratio),
		batch_size  = batch_size,
		drop_last   = True,
		# num_workers = 16, 
		num_workers = 0,
		pin_memory  = False,
		worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn = null_collate,
	)
	
 
	valid_loader = DataLoader(
		valid_dataset,
		# sampler = SequentialSampler(valid_dataset),
		batch_size  = 8,
		drop_last   = False,
		num_workers = 16,
		pin_memory  = False,
		collate_fn = null_collate,
	)

	# log.write(f'fold = {fold}\n')
	# log.write(f'ratio = {ratio}\n')
	# log.write(f'train_dataset : \n{train_dataset}\n')
	# log.write(f'valid_dataset : \n{valid_dataset}\n')
	# log.write('\n')

	# ## net ----------------------------------------
	# log.write('** net setting **\n')
	
	scaler = amp.GradScaler(enabled = is_amp)
	net = Net().cuda()

	if initial_checkpoint is not None:
		f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
		start_iteration = f['iteration']
		start_epoch = f['epoch']
		state_dict  = f['state_dict']
		net.load_state_dict(state_dict,strict=False)  #True
	else:
		start_iteration = 0
		start_epoch = 0
		net.load_pretrain()




	# log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
	# log.write('\n')


	## optimiser ----------------------------------
	if 0: ##freeze
		for p in net.encoder.parameters():   p.requires_grad = False
		for p in net.decoder.parameters():   p.requires_grad = False
		pass

	def freeze_bn(net):
		for m in net.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
				m.weight.requires_grad = False
				m.bias.requires_grad = False
	#freeze_bn(net)

	#-----------------------------------------------

	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
	# optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
	
	
	# log.write('optimizer\n  %s\n'%(optimizer))
	# log.write('\n')
	print('optimizer\n  %s\n'%(optimizer))
	print('\n')
 
	num_iteration = num_epoch*len(train_loader)
	iter_log   = int(len(train_loader)*1) #479
	iter_valid = iter_log
	iter_save  = iter_log
 
	## start training here! ##############################################
	#array([0.57142857, 0.42857143])
	# log.write('** start training here! **\n')
	# log.write('   batch_size = %d,  ratio = %d\n'%(batch_size, ratio))
	# log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
	# log.write('                           |------------------ VALID--------------|---- TRAIN/BATCH --------------------\n')
	# log.write('rate     iter        epoch | loss   auc    pfb     max_pfb        | loss                 | time         \n')
	# log.write('--------------------------------------------------------------------------------------------------------\n')
	# 		  #6.70e-4   00002856*   2.00 | 0.110  0.699  0.0858  0.134  0.158   | 0.210  0.000  0.000  |  0 hr 08 min

	print('** start training here! **\n')
	print('   batch_size = %d,  ratio = %d\n'%(batch_size, ratio))
	print('   experiment = %s\n' % str(__file__.split('/')[-2:]))
	print('                           |------------------ VALID--------------|---- TRAIN/BATCH --------------------\n')
	print('rate     iter        epoch | loss   auc    pfb     max_pfb        | loss                 | time         \n')
	print('--------------------------------------------------------------------------------------------------------\n')
 
 
	def message(mode='print'):
		asterisk = ' '
		if mode==('print'):
			loss = batch_loss
		if mode==('log'):
			loss = train_loss
			if (iteration % iter_save == 0): asterisk = '*'
		
		text = \
			('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
			'%4.3f  %4.3f  %4.4f  %4.3f  %4.3f  %4.3f  %4.3f  | '%(*valid_loss,) + \
			'%4.3f  %4.3f  %4.3f  | ' % (*loss,) + \
			'%s' % ((time.time() - start_timer))
		
		return text

	#----
	valid_loss = np.zeros(7,np.float32)
	train_loss = np.zeros(3,np.float32)
	batch_loss = np.zeros_like(train_loss)
	sum_train_loss = np.zeros_like(train_loss)
	sum_train = 0
	

	start_timer = time.time()
	iteration = start_iteration
	epoch = start_epoch
	rate = 0
	while iteration < num_iteration:
		for t, batch in enumerate(train_loader):
			
			if iteration%iter_save==0:
				if epoch < skip_save_epoch:
					n = 0
				else:
					n = iteration

				if iteration != start_iteration:
					torch.save({
						'state_dict': net.state_dict(),
						'iteration': iteration,
						'epoch': epoch,
					}, f'{fold_dir}/checkpoint/{n:08d}.model.pth')
					pass
			
			
			if (iteration%iter_valid==0): # or (t==len(train_loader)-1):
				#if iteration!=start_iteration:
				#if debug_mode == 'view-valid':
				valid_loss = do_valid(net, valid_loader)  #
				pass
			
			
			if (iteration%iter_log==0) or (iteration%iter_valid==0):
				print('\r', end='', flush=True)
				# log.write(message(mode='log') + '\n')
				
			# learning rate schduler ------------
			# adjust_learning_rate(optimizer, scheduler(epoch))
			# rate = get_learning_rate(optimizer)[0] #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)
			
			# one iteration update  -------------
			batch_size = len(batch['index'])
			for k in tensor_key: batch[k] = batch[k].cuda()

			net.train()
			net.output_type = ['loss', 'inference']
			#with torch.autograd.set_detect_anomaly(True):
			if 1:
				with amp.autocast(enabled = is_amp):
					output = net(batch)
					# output = data_parallel(net,batch)
					loss0 = output['cancer_loss'].mean()
					#loss1 = output['vindr_birads_loss'].mean()

				optimizer.zero_grad()
				scaler.scale(loss0).backward()

				scaler.unscale_(optimizer)
				#torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
				scaler.step(optimizer)
				scaler.update()
				#

			# print statistics  --------
			batch_loss[:3] = [loss0.item(), 0, 0]
			sum_train_loss += batch_loss
			sum_train += 1
			if t % 100 == 0:
				train_loss = sum_train_loss / (sum_train + 1e-12)
				sum_train_loss[...] = 0
				sum_train = 0
			
			print('\r', end='', flush=True)
			print(message(mode='print'), end='', flush=True)
			epoch += 1 / len(train_loader)
			iteration += 1
			
			# debug  --------
			#show_cam_heatmap(batch, output, net, wait=0)

			# if debug_mode == 'view-train':
			# 	show_result(batch, output, wait=0)
			# else:
			# 	if iteration%50==0:
			# 		show_result(batch, output, wait=1)
				
		# torch.cuda.empty_cache()
  
	# log.write('\n')

# main #################################################################
if __name__ == '__main__':
    
###############################################################
##### step1: main()
###############################################################
	run_train()

'''
 

'''
