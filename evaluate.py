#coding=utf-8
import numpy as np
from PIL import Image
from scipy.io import loadmat as sio
import pdb  # bebug


def IOU(lab, res, im):
	cla_num = [0] * 21
	iou = [0] * 21
	for r in range(21):
		n = np.sum(lab == r)  # array 2 list, then count numbers of the cur eme.
		if n != 0:  # means has the r-ed class
			# pdb.set_trace() 
			cla_num[r] = 1
			n_res = np.sum(res == r)
			if n_res == 0:
				iou[r] = 0
			else:
				J = 0
				for i in range(lab.shape[0]):
					for j in range(lab.shape[1]):
						if res[i,j] == r and lab[i,j] == r:
							J += 1
				# print n, n_res, J, im[:-1], 'class=', r
				iou[r] = J / (n + n_res - J + 1e-9)
	print iou, cla_num
	return iou, cla_num





lab_dir = "/path to gt /"   
label_file = "/path to /val_id.txt"
# only need change res label dir
res_dir = "/path net result label/"  #.mat
ims = open(label_file)
ious =  [[0 for col in range(21)] for row in range(1449)]   # [1449, 21]
clsaaes = [[0 for col in range(21)] for row in range(1449)]
idx = 0
for im in ims:
	lab = Image.open(lab_dir + im[:-1] + ".png")
	mat_file=sio(res_dir + im[:-1] + "_blob_0")
	mat_file=mat_file['data']
	res=np.argmax(mat_file,axis=2).astype(np.uint8)
	lab = np.array(lab)
	# 表示这里很坑, Image.open读图会把图像的size h w 互换了......  手动转回来.....
	lab = np.transpose(lab, [1,0])
	# print lab.shape
	height = min(res.shape[0], lab.shape[0])
	width = min(res.shape[1], lab.shape[1])
	res = res[:height, :width, 0]
	lab = lab[:height, :width]
	# print lab.shape, res.shape, im[:-1]
	iou, cla_num = IOU(lab, res, im)
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
