#coding=utf-8
import numpy as np
from PIL import Image
from scipy.io import loadmat as sio


def IOU(lab, res):
	cla_num = [0] * 21
	iou = [0] * 21
	for r in range(21):
		n = str(lab.tolist()).count(str(r))  # array 2 list, then count numbers of the cur eme.
		if n != 0:  # means has the r-ed class
			cla_num[r] = 1
			n_res = str(res.tolist()).count(str(r))
			if n_res == 0:
				iou[r] = 0
				continue
			else:
				J = 0
				for i in range(lab.shape[0]):
					for j in range(lab.shape[1]):
						# be careful of lab col_lab, res     col_lab and res is colorful, lab is 0~20 
						if res[i, j] == lab[i,j] and lab[i,j] == r:
							J += 1
				iou[r] = J / (n + n_res - J + 0.0000001)
	print iou, cla_num
	return iou, cla_num


lab_dir = "/path to gt label/"     
label_file = "/path to val_id.txt/"
# only need change res label dir
res_dir = "/path to net res mat file/"  #.mat
ims = open(label_file)
ious =  [[0 for col in range(21)] for row in range(1449)]   # [1449, 21]
clsaaes = [[0 for col in range(21)] for row in range(1449)]
idx = 0
for im in ims:
	# print res_dir + im[:-1] + "_blob_0.png"
	lab = Image.open(lab_dir + im[:-1] + ".png")
	mat_file=sio(res_dir + im[:-1] + "_blob_0")
	mat_file=mat_file['data']
	res=np.argmax(mat_file,axis=2).astype(np.uint8)
	res = res.reshape(513, 513)  # 0~20 label data
	# resize
	lab = lab.resize(res.shape)
	lab = np.array(lab)
	iou, cla_num = IOU(lab, res)
	ious[idx] = iou
	clsaaes[idx] = cla_num
	idx += 1

iou = map(sum, zip(*ious))  # sum in col
print iou
cla = map(sum, zip(*clsaaes))
print cla

res = 0
for i in range(21):
	res += iou[i] / cla[i]
print res / 21








