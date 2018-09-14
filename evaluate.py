#coding=utf-8
import numpy as np
from PIL import Image
 
def IOU(lab, res):
	iou = [0] * 21
	for r in range(21):
		n = str(lab.tolist()).count(str(r))  # array 2 list, then count numbers of the cur eme.
		if n != 0:  # means has the r-ed class 
			n_res = str(res.tolist()).count(str(r))
			if n_res == 0:
				iou[r] = 0
				continue
			else:
				J = 0
				for i in range(lab.shape[0]):
					for j in range(lab.shape[1]):
						if res[i, j] == lab[i,j] and res[i,j] == r:
							J += 1
				iou[r] = J / (n + n_res - J + 0.0000001)
	return iou
 
 
lab_dir = "/path to gt/"
label_file = "/path to val_id.txt/"
# only need change res label dir
res_dir = "/path to net result label/"
ims = open(label_file)
ious = []  # 1499 iou lists
for im in ims:
	# print res_dir + im[:-1] + "_blob_0.png"
	lab = Image.open(lab_dir + im[:-1] + ".png")
	res = Image.open(res_dir + im[:-1] + "_blob_0.png")
	# resize
	lab = lab.resize(res.size)
	res = res.resize(res.size)
	lab = np.array(lab)
	res = np.array(res)
	ious.append(IOU(lab, res))
 
miou = [0] * 21
for iou in ious:
	for i in range(21):
		miou[i] += iou[i]
MIOU = sum(miou) / (1499 * 21)
print MIOU
 
 
 
