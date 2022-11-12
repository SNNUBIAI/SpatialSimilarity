import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from metrics.utils import Masker, padding_1d

def IoU(n1, n2):
	"""
	:param n1: 1*N
	:param n2: 1*N
	:return: IoU
	"""
	intersect = np.logical_and(n1, n2)
	union = np.logical_or(n1, n2)
	I = np.count_nonzero(intersect)
	U = np.count_nonzero(union)
	return I / U

class SpatialSimilarity:
	def __init__(self, mask_img=None):
		if mask_img is not None:
			self.masker = Masker(mask_path=mask_img)
		else:
			self.masker = None
		self.local_3d_kernel = None
		self.local_2d_kernel = None

	def local_IoU_1d(self, n1, n2, window_size=3, padding=1):
		n1 = padding_1d(n1, padding=padding)
		n2 = padding_1d(n2, padding=padding)

		intersect = 0
		cnt_1_ls = []
		cnt_2_ls = []
		for i in range(0, n1.shape[1]):
			t1 = n1[:, i:i+window_size]
			t2 = n2[:, i:i+window_size]

			cnt_1 = np.count_nonzero(t1)
			cnt_2 = np.count_nonzero(t2)
			cnt_1_ls.append(cnt_1)
			cnt_2_ls.append(cnt_2)
			if cnt_1 > 0 and cnt_2 > 0:
				intersect += 1
		union = np.logical_or(np.array(cnt_2_ls), np.array(cnt_1_ls))
		union = np.count_nonzero(union)
		return intersect / union

	@torch.no_grad()
	def local_IoU_2d(self, n1, n2, window_size=3, padding=1):
		"""
		:param n1: (1*1*H*W)
		:param n2: (1*1*H*W)
		:return: local IoU
		"""
		self.local_2d_kernel = torch.tensor(np.ones((window_size, window_size)),
											dtype=torch.float,
											requires_grad=False).view(1, 1, window_size, window_size)
		n1 = torch.tensor(n1, dtype=torch.float)
		n2 = torch.tensor(n1, dtype=torch.float)
		cnt_1 = F.conv2d(n1, self.local_2d_kernel, padding=(padding, padding), stride=1).detach().cpu().numpy()
		cnt_2 = F.conv2d(n2, self.local_2d_kernel, padding=(padding, padding), stride=1).detach().cpu().numpy()
		return IoU(cnt_1.flatten().reshape(1, -1), cnt_2.flatten().reshape(1, -1))
		
	@torch.no_grad()
	def local_IoU_3d(self, n1, n2, window_size=3, padding=1):
		self.local_3d_kernel = torch.tensor(np.ones((window_size, window_size, window_size)),
											dtype=torch.float,
											requires_grad=False).view(1, 1,
																	  window_size, window_size, window_size)
		n1_3d = self.masker.inverse_transform2tensor(n1).unsqueeze(1)
		n2_3d = self.masker.inverse_transform2tensor(n2).unsqueeze(1)
		cnt_1 = F.conv3d(n1_3d, self.local_3d_kernel, padding=(padding, padding, padding), stride=1).squeeze(1)
		cnt_2 = F.conv3d(n2_3d, self.local_3d_kernel, padding=(padding, padding, padding), stride=1).squeeze(1)
		
		cnt_1 = self.masker.tensor_transform(cnt_1)
		cnt_2 = self.masker.tensor_transform(cnt_2)
		
		return IoU(cnt_1, cnt_2)