# from common import *

#from timm.models.resnet import *
from timm.models.efficientnet import *
#from coat import *
from configure import *

import torch
import torch.nn.functional as F
import torch.nn as nn


##################################################################################33
def criterion_cross_entropy(logit, truth):
	num_class = logit.shape[-1]
	return F.cross_entropy(logit.reshape(-1,num_class), truth.reshape(-1))


## different head for aux loss
#<todo> ignore index
class VindrHead(nn.Module):

	def loss(self, output, target):

		# output['vindr_abnormality_loss'] = F.binary_cross_entropy_with_logits(
		# 	output['vindr_abnormality'][target['vindr_index']],
		# 	target['vindr_abnormality']
		# )
		output['vindr_density_loss'] = criterion_cross_entropy(
			output['vindr_density'],
			target['vindr_density']
		)
		output['vindr_birads_loss'] = criterion_cross_entropy(
			output['vindr_birads'],
			target['vindr_birads']
		)
		return output

	def inference(self, output):
		#output['vindr_abnormality'] = torch.sigmoid(output['vindr_abnormality'])
		output['vindr_density'] = torch.softmax(output['vindr_density'].float(),-1)
		output['vindr_birads'] = torch.softmax(output['vindr_birads'].float(),-1)
		return output

	#-----
	def __init__(self, in_channel):
		super(VindrHead, self).__init__()

		#self.heatmap     = nn.Linear(in_channel, 5)
		#self.abnormality = nn.Linear(in_channel, 5)
		self.density     = nn.Linear(in_channel, 4)
		self.birads      = nn.Linear(in_channel, 7)

	def forward(self, x):
		#heatmap = self.heatmap(feature)
		#heatmap = rearrange(heatmap, 'b v (h w) c -> b v c h w ', h=image_height//heatmap_scale, w=image_width//heatmap_scale)
		# cancer = self.cancer(cls)

		#abnormality = self.abnormality(cls)
		density = self.density(x)
		birads = self.birads(x)

		output = {
			#'vindr_abnormality_heatmap': heatmap,
			#'vindr_abnormality': abnormality,
			'vindr_density': density,
			'vindr_birads': birads,
		}
		return output





# https://keras.io/examples/vision/vit_small_ds/
# https://github.com/lucidrains/vit-pytorch/blob/main/README.md
# jitter (Shifted Patch Tokenization) trick to improve accuracy (especially for small object)
#
def jitter(x):
	#shift = [[1, -1, 0, 0] , [0, 0, 1, -1]]
	shift = [[2, -2, 0, 0] , [0, 0, 2, -2]]
	xx=[x]
	for s in shift:
		xx.append(
			F.pad(x, s, mode='constant', value=0)
		)
	xx = torch.cat(xx, 1)
	return xx



pretrain_dir= '/home/titanx/hengck/share1/data/pretrain_model'
class Net(nn.Module):
	def load_pretrain(self, ):
		pass

	def __init__(self,):
		super(Net, self).__init__()
		self.output_type = ['inference', 'loss']
		self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
		self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))

		self.encoder = efficientnet_b0(pretrained=True, drop_rate=0.2, drop_path_rate=0.2)

		#aux head
		self.cancer = nn.Linear(1280,1)
		#self.vindr = VindrHead(1792)


	def forward(self, batch):
		x = batch['image']
		batch_size,C,H,W = x.shape


		#---
		#x = jitter(x)
		x = (x - self.mean) / self.std


		#------
		#single view
		e = self.encoder.forward_features(x)
		x = F.adaptive_avg_pool2d(e,1)
		x = torch.flatten(x,1,3)


		# aux head ----
		#vindr = self.vindr(x)
		cancer = self.cancer(x).reshape(-1)


		output = {}
		if  'loss' in self.output_type:
			pass
			# r = self.vindr.loss(vindr, batch)
			# output = {**output,  **r }
			output['cancer_loss']=F.binary_cross_entropy_with_logits(cancer,batch['cancer'])

		if 'inference' in self.output_type:
			pass

			w = self.cancer.weight.reshape(1,-1, 1, 1)
			cam = torch.clamp((e * w).sum(1), min=0)

			output['cancer']=torch.sigmoid(cancer)
			output['cam']=cam
			# r = self.vindr.inference(vindr)
			# output = {**output, **r}

		return output


def run_check_net():

	h,w = image_height, image_width
	batch_size = 4

	# ---
	batch = {
		'image' : torch.from_numpy(np.random.uniform(0,1,(batch_size, 1, h,w))).float().cuda(),
		#'part'  : torch.from_numpy(np.random.uniform(0,1,(batch_size,2, 5, h//heatmap_scale,w//heatmap_scale))).float().cuda(),
		'cancer': torch.from_numpy(np.random.choice(2,(batch_size))).float().cuda(),

		#'vindr_index' : [0,1,2,3],
		#'vindr_abnormality_heatmap' : torch.from_numpy(np.random.uniform(0,1,(batch_size, 2, 5, h//heatmap_scale,w//heatmap_scale))).float().cuda(),
		#'vindr_abnormality' : torch.from_numpy(np.random.uniform(0,1,(batch_size, 2, 5))).float().cuda(),
		#'vindr_density'     : torch.from_numpy(np.random.choice(4,(batch_size))).long().cuda(),
		#'vindr_birads'      : torch.from_numpy(np.random.choice(7,(batch_size))).long().cuda(),
	}
	#batch = {k: v.cuda() for k, v in batch.items()}

	net = Net().cuda()
	net.load_pretrain()
	#net.output_type = ['inference']#, 'loss']

	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)

	print('batch')
	for k, v in batch.items():
		if 'index' in k: continue
		print('%32s :' % k, v.shape)

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	print('')
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	run_check_net()