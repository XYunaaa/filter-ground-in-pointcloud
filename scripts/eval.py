import os
import glob
import time
import numpy as np
import KITTIReader as reader
import utils
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
        '''
        raw_data_copy = raw_data.copy()
        raw_data_copy = raw_data_copy[:,:3]
        ground_index_ = []
        non_ground_index_ = []

        for i in range(len(points_down)):
            p = points_down[i][0:3]
            index = np.where(raw_data_copy == p)[0]
            ground_index_.append(index[0])
            print(ground_index[i],index[0])

        for i in range(len(points_up)):
            p = points_up[i][0:3]
            index = np.where(raw_data_copy == p)[0]
            non_ground_index_.append(index[0])
        '''
        points_down_labels = points_down[:,3]
        points_up_labels = points_up[:,3]

    else:
        print('Method have not release')
        return None,None,None,None,0

    return points_up, points_up_labels, points_down, points_down_labels, t

def cal(raw_path):

    file = os.listdir(raw_path)
    file_list = []
    for f in file:
        if ".bin" in f:
            file_list.append(f)

    file_list = sorted(file_list)
    raw_points_count = 0
    non_ground_count = 0
    for velo_file in file_list:
        num = velo_file.split('.')[0]
        if int(num) % 50 == 0:
            print('now tracking vel:', velo_file)
        raw_data = reader.load_velo_scan(raw_path + '/' + velo_file)
        raw_label = np.zeros((raw_data.shape[0],1))
        fileter_height, _, _, _, _ = filter_ground(raw_data, raw_label,4)
        raw_points_count +=raw_data.shape[0]
        non_ground_count +=fileter_height.shape[0]
    print('non_ground pointclouds/all pointclouds: ',non_ground_count/raw_points_count)


raw_path = '../data/velodyne/'
cal(raw_path) # The ratio of (non-ground pointclouds/all pointclouds)

# calculate the performance of each method in SemanticKITTI(00-10 sequences)
for method in range(0,5):
    if method == 0:
        method_name = 'height_diff/'
    elif method == 1:
        method_name = 'linefit/'
    elif method == 2:
        method_name = 'PlaneFit_Opitimizer/'
    elif method == 3:
        method_name = 'PlaneFit_RANSAC/'
    else:
        method_name = 'PCL-Rand/'

    print('method',method_name)
    ground_error_method = 0
    car_error_method = 0
    ped_error_method = 0
    cyc_error_method = 0
    N_method = 0
    velo_count_method = 0
    error_method = 0
    accuracy_method = 0
    precision_method = 0
    recall_method = 0
    miou_method = 0
    Time_method = 0
    for seq in range(0,11):

        print('Now tracking Seq:',seq)
        ground_error = 0
        car_error = 0
        ped_error = 0
        cyc_error = 0
        N = 0
        velo_count = 0
        precision = 0
        recall = 0
        miou = 0
        Time = 0

        infile_path = '/media/ddd/data1/vo/sequences_0.06/'+str(seq).zfill(2)+'/velodyne'
        label_path = '/media/ddd/data1/vo/sequences_0.06/'+str(seq).zfill(2)+'/labels/'
        output_path = 'result/'+ method_name

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_path = output_path+str(seq).zfill(2) + '.txt'
        file = os.listdir(infile_path)
        file_list = []
        for f in file:
            if ".npy" in f:
                file_list.append(f)
        file_list = sorted(file_list)

        for velo_file in file_list:
            num = velo_file.split('.')[0]
            if int(num) <= 100:
                print(velo_file)
                raw_data = reader.load_velo_scan(infile_path + '/'+velo_file)
                raw_label = np.load(label_path + velo_file)
                #Plot.draw_pc_sem_ins(raw_data, raw_data[:,-2])
                # ret_nonGroundPoints, ret_nonGroundLabels,GroundPoints, GroundLabels, elapsed
                fileter_height,fileter_height_label,ground,fileter_height_label_ground,t = filter_ground(raw_data,raw_label,method)

                #print(fileter_height.shape[0],raw_data.shape[0])
                #Plot.draw_pc_sem_ins(fileter_height,fileter_height_label)
                #Plot.draw_pc_sem_ins(ground[:,:3], fileter_height_label_ground)
                filter_label = fileter_height_label.reshape((-1,1))
                filter_label_ground = fileter_height_label_ground.reshape((-1,1))
                miou_temp,TP_temp,FP_temp,FN_temp,TN_temp,pre,rec, car_error_temp, ped_error_temp, cyc_error_temp,ground_error_temp = eval(raw_label, filter_label, filter_label_ground)
                #print(ground_precision_temp, car_error_temp, ped_error_temp, err, acc, pre, rec)
                N += 1
                velo_count += 1
                #ground_precision += ground_precision_temp
                car_error += car_error_temp
                ped_error += ped_error_temp
                cyc_error += cyc_error_temp
                ground_error += ground_error_temp
                #accuracy += acc
                precision += pre
                recall += rec
                miou += miou_temp
                Time += t

        print('ground miou :',miou/N)
        print('car_error :', car_error/N)
        print('ped_error :', ped_error/N)
        print('cyc_error :', cyc_error/velo_count)
        print('ground_error :', ground_error / velo_count)
        #print('accuracy :',accuracy/velo_count)
        print('precision :',precision/velo_count)
        print('recall :',recall/velo_count)
        print('time :',Time/N)
        N_method += N
        velo_count_method += velo_count
        #ground_precision_method += ground_precision
        car_error_method += car_error
        ped_error_method += ped_error
        ground_error_method += ground_error
        cyc_error_method += cyc_error
        #accuracy_method += accuracy
        precision_method += precision
        recall_method += recall
        miou_method += miou
        Time_method += Time

        with open(output_path, "w") as f:

            f.write("ground_miou :"+str(miou/N)+"\n")
            f.write("car_error :"+str(car_error/N)+"\n")
            f.write("ped_error :" + str(ped_error / N) + "\n")
            f.write("cyc_error :" + str(cyc_error / N) + "\n")
            f.write("ground_error :" + str(ground_error / N) + "\n")
            f.write("precision: "+str(precision/velo_count)+"\n")
            f.write("recall: "+str(recall/velo_count)+"\n")
            f.write("time: "+str(Time/N)+"\n")


        #eval(raw_label, fileter_linefit_labels)

        #utils.plot_pointClouds(preProcess.get_rawPointCloud())
        #utils.plot_pointClouds(preProcess.ground_filter_linefit(raw_data, 15))
        #utils.plot_pointClouds(preProcess.ground_filter_heightDiff(raw_data, 400, 400, 0.3, 0.5))'''


    output_method_path = 'result/'+method_name+'/total.txt'

    with open(output_method_path, "w") as f:
        f.write("ground_miou :" + str(miou_method / N_method) + "\n")

        f.write("car_error :" + str(car_error_method / N_method) + "\n")
        f.write("ped_error :" + str(ped_error_method / N_method) + "\n")
        f.write("cyc_error :" + str(cyc_error_method / N_method) + "\n")
        f.write("ground_error: " + str(ground_error_method / velo_count_method) + "\n")

        f.write("precision: " + str(precision_method/ velo_count_method) + "\n")
        f.write("recall: " + str(recall_method / velo_count_method) + "\n")
        f.write("time: " + str(Time_method / N_method) + "\n")


