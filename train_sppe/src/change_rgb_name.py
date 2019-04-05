import os
from IPython import embed
img_base = "/media/data_2/COCO_SIXD/linemod_test_kp/01/kp_label"
img_files = os.listdir(img_base)
for idx, temp in enumerate(img_files):
	# print temp
	num = temp.rfind('.')
	if num != -1:	
		new_name = int(temp[: num])
		# embed()
		new_name = "%012d" % new_name
		new_name = new_name + ".npy"
		if idx%1000 == 0: 
			print(idx, "finished!")
		os.rename(os.path.join(img_base, temp),os.path.join(img_base, new_name))
	# embed()
	# break