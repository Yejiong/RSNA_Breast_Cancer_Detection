from configure import *
from augmentation import *

# from common import *
#from natsort import natsorted
from torch.utils.data import Dataset
import pdb
import torch
import math
import pandas as pd
import cv2
import numpy as np


image_dir = "D:/RSNA_data/"


############################################################################################

def read_kaggle_csv():
	df = pd.read_csv('./my_split/valid_df.fold0.csv')
	valid_id = df.patient_id.unique()

	csv_file = './my_split/kaggle.ver01.csv'
	kaggle_df = pd.read_csv(csv_file)
	patient_id = kaggle_df.patient_id.unique()
	train_id =list(set(patient_id)-set(valid_id))

	train_df = kaggle_df[kaggle_df.patient_id.isin(train_id)].reset_index(drop=True)
	valid_df = kaggle_df[kaggle_df.patient_id.isin(valid_id)].reset_index(drop=True)
	return train_df, valid_df


train_df, valid_df = read_kaggle_csv()


def paste_into_by_center(dst, src):
	dh, dw = dst.shape[:2]
	sh, sw = src.shape[:2]
	x = (dw - sw) // 2
	y = (dh - sh) // 2
	dst[y:y + sh, x:x + sw] = src
	return dst


def read_kaggle_data(d):
	
	# image = cv2.imread(f'{image_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png', cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(f'{image_dir}/{d.patient_id}_{d.image_id}.png', cv2.IMREAD_GRAYSCALE)
 
	h, w = image.shape

	xmin, ymin, xmax, ymax = (np.array(eval(d.pad_breast_box)) * h).astype(int)
	crop = image[ymin:ymax, xmin:xmax]

	mh, mw = (np.array(eval(d.max_pad_breast_shape)) * h).astype(int)
	#scale = image_height / h
	scale = mh / h
	if (scale*mw) > image_width:
		scale = image_width/mw

	dsize = ( min(image_width, int(scale*crop.shape[1])), min(image_height, int(scale*crop.shape[0])) )
	crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
	#print(crop.shape)

	data = np.zeros((image_height,image_width), np.uint8)
	data = paste_into_by_center(data, crop)

	data = data/255
	return {
		'image': data,
		'cancer': d.cancer,
	}


############################################################################################

class RsnaDataset(Dataset):
	def __init__(self, df, augment=None):

		self.df = df
		self.augment = augment
		self.length = len(df)

	def __str__(self):

		num_patient = len(set(self.df.patient_id))
		num_image = len(self.df)

		string = ''
		string += f'\tlen = {len(self)}\n'
		string += f'\tnum_patient = {num_patient}\n'
		string += f'\tnum_image = {num_image}\n'

		count = dict(self.df.cancer.value_counts())
		for k in [0,1]:
			string += f'\t\tcancer{k} = {count.get(k,0):5d} ({count.get(k,0)/len(self.df):0.3f})\n'
		return string

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		d = self.df.iloc[index]


		################################################
		e = read_kaggle_data(d)

		if self.augment is not None:
			e = self.augment(e)



		#---------------------------
		r = {}
		r['index'] = index
		r['d'] = d
		r['image' ] = torch.from_numpy(e['image']).float()
		r['cancer' ] = torch.FloatTensor([e['cancer']])
		# r['vindr_birads'  ] = torch.LongTensor([e['birads']])
		return r






tensor_key = ['image', 'cancer']
def null_collate(batch):
	batch_size = len(batch)
	d = {}
	key = batch[0].keys()
	for k in key:
		v = [b[k] for b in batch]
		d[k] = v

	d['image']= torch.stack(d['image'],0).unsqueeze(1)
	d['cancer']= torch.cat(d['cancer'],0)
	return d


# class BalanceSampler(Sampler):

# 	def __init__(self, dataset, ratio=8):
# 		self.r = ratio-1
# 		self.dataset = dataset
# 		self.pos_index = np.where(dataset.df.cancer>0)[0]
# 		self.neg_index = np.where(dataset.df.cancer==0)[0]

# 		N = int(np.floor(len(self.neg_index)/self.r))
# 		self.neg_length = self.r*N
# 		self.pos_length = N
# 		self.length = self.neg_length + self.pos_length


# 	def __iter__(self):
# 		pos_index = self.pos_index.copy()
# 		neg_index = self.neg_index.copy()
# 		np.random.shuffle(pos_index)
# 		np.random.shuffle(neg_index)

# 		neg_index = neg_index[:self.neg_length].reshape(-1,self.r)
# 		pos_index = np.random.choice(pos_index, self.pos_length).reshape(-1,1)
# 		index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
# 		return iter(index)

# 	def __len__(self):
# 		return self.length

#################################################################################

def train_augment_v00(e):
	image = e['image']

	#image = do_random_hflip(image) # hflip, vflip or both
	image = do_random_flip(image)

	if np.random.rand() < 0.7:
		for func in np.random.choice([
			lambda image : do_random_affine( image, degree=0, translate=0.005, scale=0.1, shear=20),
			lambda image : do_random_rotate(image,  degree=20),
			lambda image : do_random_stretch(image, stretch=(0.2,0.2)),
			lambda image : do_elastic_transform(
				image,
				alpha=image_height,
				sigma=image_height* 0.05,
				alpha_affine=image_height* 0.03
			),

		], 1):
			image = func(image)



	if np.random.rand() < 0.5:
		for func in np.random.choice([
			lambda image: do_random_contrast(image, mul=[-0.3,0.3],pow=[-0.5,0.5],add=[-0.3,0.3]),
			#lambda image: do_random_noise(image, m=0.08),
		], 1):
			image = func(image)
			pass

	#
	# if np.random.rand() < 0.25:
	# 	image = do_random_cutout(
	# 		image, num_block=5,
	# 		block_size=[0.1,0.3],
	# 		fill='constant'
	# 	)
	#

	e['image'] = image
	return e

#################################################################################

def run_check_dataset():

	train_df, valid_df = read_kaggle_csv()
	dataset = RsnaDataset(train_df, augment=train_augment_v00)
	print(dataset)

	for i in range(12):
	#for i in range(len(dataset)):
		i = 1 #240*8+ i#np.random.choice(len(dataset))
		r = dataset[i]
		print(r['index'], 'id = ', r['d']['patient_id'], '-----------')
		for k in tensor_key :
			v = r[k]
			print(k)
			print('\t', 'dtype:', v.dtype)
			print('\t', 'shape:', v.shape)
			if len(v)!=0:
				print('\t', 'min/max:', v.min().item(),'/', v.max().item())
				print('\t', 'is_contiguous:', v.is_contiguous())
				print('\t', 'values:')
				print('\t\t', v.reshape(-1)[:8].data.numpy().tolist(), '...')
				print('\t\t', v.reshape(-1)[-8:].data.numpy().tolist())
		print('')
		if 1:
			image = r['image'].data.cpu().numpy()

			#image_show_norm('image'+f'{i}', image, resize=0.5)
			image_show_norm('image', image, resize=0.5)
			cv2.waitKey(0)



	loader = DataLoader(
		dataset,
		sampler=SequentialSampler(dataset),
		#sampler=BalanceSampler(dataset),
		batch_size=8,
		drop_last=True,
		num_workers=0,
		pin_memory=False,
		worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn=null_collate,
	)
	print(loader.batch_size, len(loader), len(dataset))
	print('')

	for t, batch in enumerate(loader):
		if t > 5: break
		print('batch ', t, '===================')
		print('index', batch['index'])
		for k in tensor_key:
			v = batch[k]
			print(k)
			print('\t', 'shape:', v.shape)
			print('\t', 'dtype:', v.dtype)
			print('\t', 'min/max:', v.min().item(),'/', v.max().item())
			print('\t', 'is_contiguous:', v.is_contiguous())
			print('\t', 'value:')
			print('\t\t', v.reshape(-1)[:8].data.numpy().tolist())
			# if k=='cancer':
			# 	print('\t\tsum ', v.sum().item())
		if 1:
			pass
			#print('vindr_index:')
			#print('\t',batch['vindr_index'])
		print('')





def run_check_augment():

	train_df, valid_df = read_vindr_csv()
	dataset = RsnaDataset(train_df)
	print(dataset)

	#---------------------------------------------------------------
	def augment(image):
		#image = do_random_flip(image)

		#image = do_random_affine( image, degree=0, translate=0, scale=0.1, shear=20)
		#image = do_random_rotate(image,  degree=20)
		#image = do_random_stretch(image, stretch=(0.2,0.2))

		# image = do_elastic_transform(
		# 	image,
		# 	alpha=image_height,
		# 	sigma=image_height * 0.05,
		# 	alpha_affine=image_height * 0.03
		# )

		image = do_random_contrast(image)
		return image

	for i in range(10):
		#i = 2424 #np.random.choice(len(dataset))#272 #2627
		print(i)
		r = dataset[i]

		image  = r['image'].data.cpu().numpy()
		image_show_norm('image',image, min=0, max=1,resize=1)
		#cv2.waitKey(0)

		for t in range(100):
			image1 = augment(image.copy())
			image_show_norm('image1', image1, min=0, max=1,resize=1)
			cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
	run_check_dataset()
	#run_check_augment()