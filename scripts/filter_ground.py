import time
import numpy as np
import KITTIReader as reader
from preProcessor import preProcessor
from utils import Plot,eval
import plane_fit
import ground_removal_ext

def filter_ground(raw_data, raw_label,method_num):

    if method_num==0:
        preProcess = preProcessor()
        points_up, points_up_labels, points_down, points_down_labels, t = preProcess.ground_filter_heightDiff(raw_data, raw_label, 400, 400, 0.3, 0.005)
    elif method_num == 1:
        preProcess = preProcessor()
        points_up, points_up_labels, points_down, points_down_labels, t  = preProcess.ground_filter_linefit(raw_data, raw_label, 15)
    elif method_num ==2:
        points_up, points_down, non_ground_index, ground_index, t = plane_fit.get_fit_plane_LSE_RANSAC_Opitimizer(raw_data)
        points_up_labels = raw_label[non_ground_index]
        points_down_labels = raw_label[ground_index]
    elif method_num == 3:
        points_up, points_down, non_ground_index, ground_index, t = plane_fit.get_fit_plane_LSE_RANSAC(raw_data)
        points_up_labels = raw_label[non_ground_index]
        points_down_labels = raw_label[ground_index]
    elif method_num==4:
        t1 = time.time()
        raw_data[:,3] = raw_label.reshape(-1)
        segmentation = ground_removal_ext.ground_removal_kernel(raw_data, 0.2, 200)  # distance_th=0.2, iter=200
        t2 = time.time()
        t = t2-t1
        non_ground_index = np.where(segmentation[:, -1] == 0)[0]
        points_up = segmentation[non_ground_index][:, :]
        ground_index = np.where(segmentation[:, -1] != 0)[0]
        points_down = segmentation[ground_index][:, :]
        points_down_labels = points_down[:,3]
        points_up_labels = points_up[:,3]

    else:
        print('Method have not release')
        return None,None,None,None,0

    return points_up, points_up_labels, points_down, points_down_labels, t


have_label = True
method = 4

infile_path = '../data/velodyne/000002.npy'
raw_data = reader.load_velo_scan(infile_path)

if have_label:
    label_path = '../data/label/000002.npy'
    raw_label = np.load(label_path)
else:
    raw_label = np.zeros((raw_data.shape[0],1)) +1


fileter_height,fileter_height_label,ground,fileter_height_label_ground,t = filter_ground(raw_data,raw_label,method)
Plot.draw_pc_sem_ins(fileter_height,fileter_height_label)

