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

	def local_IoU_1d(self, n1, n2, window_size=3, padding=1):
		union = np.logical_or(n1, n2)
		union = np.count_nonzero(union)

		n1 = padding_1d(n1, padding=padding)
		n2 = padding_1d(n2, padding=padding)

		intersect = 0
		for i in range(0, n1.shape[1]):
			t1 = n1[:, i:i+window_size]
			t2 = n2[:, i:i+window_size]

			cnt_1 = np.count_nonzero(t1)
			cnt_2 = np.count_nonzero(t2)
			if cnt_1 > 0 and cnt_2 > 0:
				intersect += 1
		return intersect / union