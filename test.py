import numpy as np

from metrics.utils import padding_1d
from metrics.metrics import SpatialSimilarity, IoU

# n = np.random.randn(1, 5)
# print(n)
# n = padding_1d(n, padding=0)
# print(n)
# n = padding_1d(n, padding=1)
# print(n)
# n = padding_1d(n, padding=2)
# print(n)
# print(np.logical_or(n, n))
# print(np.count_nonzero(np.logical_or(n, n)))

sm = SpatialSimilarity()
n1 = np.array([[0, 0, 1, 0, 0, 0, 0]])
n2 = np.array([[0, 0, 0, 1, 0, 0, 0]])
n3 = np.array([[0, 0, 0, 0, 0, 0, 1]])
print("n1:", n1)
print("n2:", n2)
print("n3:", n3)
print("IoU n1 and n2:", IoU(n1, n2))
iou = sm.local_IoU_1d(n1, n2, window_size=3, padding=1)
print("Local IoU n1 and n2:", iou)
print("IoU n1 and n3:", IoU(n1, n3))
iou = sm.local_IoU_1d(n1, n3, window_size=3, padding=1)
print("Local IoU n1 and n3:", iou)

sm = SpatialSimilarity()
n1 = np.array([[0, 1, 1, 0, 0, 0, 0]])
n2 = np.array([[0, 0, 1, 1, 0, 0, 0]])
n3 = np.array([[0, 0, 1, 1, 0, 0, 1]])
print("n1:", n1)
print("n2:", n2)
print("n3:", n3)
print("IoU n1 and n2:", IoU(n1, n2))
iou = sm.local_IoU_1d(n1, n2, window_size=3, padding=1)
print("Local IoU n1 and n2:", iou)
print("IoU n1 and n3:", IoU(n1, n3))
iou = sm.local_IoU_1d(n1, n3, window_size=3, padding=1)
print("Local IoU n1 and n3:", iou)