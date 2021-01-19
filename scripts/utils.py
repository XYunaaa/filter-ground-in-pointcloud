import numpy as np
from mayavi import mlab
import open3d
import colorsys, random

def eval(raw_labels, filter_labels, ground_labels):
    '''
       0: 'unlabeled',
       1: 'car',
       2: 'bicycle',
       3: 'motorcycle',
       4: 'truck',
       5: 'other-vehicle',
       6: 'person',
       7: 'bicyclist',
       8: 'motorcyclist',
       9: 'road',
       10: 'parking',
       11: 'sidewalk',
       12: 'other-ground'
    '''
    filter_labels = filter_labels.reshape((-1))# after filter NON-ground labels
    ground_labels = ground_labels.reshape((-1))# after filter ground labels
    raw_labels = raw_labels.reshape((-1))
    #print(np.unique(raw_labels))

    ## Pedestrian
    raw_ped_labels = len(np.where(raw_labels == 6)[0]) #

    ## Cyclist = bicyclist + motorcyclist
    ind_cyclist = np.where(raw_labels >= 7)
    R_ped = raw_labels[ind_cyclist]
    raw_cyclist_labels = len(np.where(R_ped <= 8)[0])  #

    # car + bicycle + motorcycle #
    ind = np.where(raw_labels >= 1)
    R = raw_labels[ind]
    raw_car_labels = len(np.where(R <= 5)[0])

    # road + parking + sidewalk + other-ground #
    ind = np.where(raw_labels >= 9)
    R = raw_labels[ind]
    raw_ground_labels = len(np.where(R <= 12)[0]) + len(np.where(R == 17)[0]) # terrain

    #print('raw_labels len', raw_labels.shape[0])
    #print('raw_ground_labels len',raw_ground_labels)

    # 经过地面过滤后，被识别为 NON-ground 的点集合对应的标签
    filter_ped_labels = len(np.where(filter_labels == 6)[0])  #
    ind = np.where(filter_labels >= 7)
    R = filter_labels[ind]
    filter_cyclist_labels = len(np.where(R <= 8)[0])

    ind = np.where(filter_labels >= 1)
    R = filter_labels[ind]
    filter_car_labels = len(np.where(R <= 5)[0])

    ind = np.where(filter_labels >= 9)
    R = filter_labels[ind]
    filter_ground_labels = len(np.where(R <= 12)[0]) + len(np.where(R == 17)[0]) ## 被识别为 NON-ground 的点集合中 实际上是地面点的个数

    TN = len(filter_labels)-filter_ground_labels


    ##### calculate tp fp fn in ground label ####
    ind = np.where(ground_labels >= 9) #在被识别为地面点的点集合中，真实标签为地面点的数量--TP
    R = ground_labels[ind]
    TP = len(np.where(R <= 12)[0]) + len(np.where(R == 17)[0])
    FP = len(ground_labels) - TP #在被识别为地面点的点集合中，真实标签为非地面点的数量--FP
    #print('filter label len',filter_labels.shape[0])
    #print('FP',FP)
    FN = raw_ground_labels - TP  #所有真值为地面点的点集合中，未被识别为地面的点数 -- FN

    if raw_car_labels == 0:
        car_error = 0 # 该帧没有车辆
    else:
        car_error = abs(filter_car_labels - raw_car_labels) / raw_car_labels


    if raw_ped_labels == 0:
        ped_error = 0
    else:
        ped_error = abs(raw_ped_labels - filter_ped_labels) / raw_ped_labels

    if raw_cyclist_labels == 0:
        cyc_error = 0
    else:
        cyc_error = abs(raw_cyclist_labels - filter_cyclist_labels) / raw_cyclist_labels

    #ground_precision = abs(raw_ground_labels - filter_ground_labels) / raw_ground_labels

    '''
    print('ground_precision :', ground_precision,raw_ground_labels,filter_ground_labels)
    print('car_error :', car_error,raw_car_labels,filter_car_labels)
    if ped_error >= 0:
        print('ped_error :', ped_error,raw_ped_labels,filter_ped_labels)
    else:
        print('No ped!')
        ped_error = 0
    '''
    ground_error = FP/(raw_ground_labels)
    #accuracy = TP/(raw_ground_labels)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    miou = TP / (TP+FP+FN)

    return miou, TP,FP,FN,TN, precision, recall, car_error, ped_error ,cyc_error,ground_error

def plot_pointClouds(pointClouds):
    x = pointClouds[:, 0]  # x position of point
    y = pointClouds[:, 1]  # y position of point
    z = pointClouds[:, 2]  # z position of point
    d = pointClouds[:, 3]  # if -1000 ground
    maxD = np.max(d)
    print(maxD)
    minD = np.min(d)
    print(minD)
    #d = (70) * (d-minD)
    #d = np.where(d>40, 70, 20)
    maxD = np.max(d)
    print(maxD)
    minD = np.min(d)
    print(minD)
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z, d, mode="point", colormap='spectral', figure=fig)
    mlab.show()

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        #random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.visualization.draw_geometries([pc])

        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        '''
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        '''
        plot_colors = [[105,105,105],[25,25,112],[100,149,237],[138,238,104]] # Other-label(grey) Vehicle(dark-blue) Person(light-blue) Ground(green)
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        if plot_colors is not None:
            vehicle_labels = np.where((pc_sem_ins>=1)&(pc_sem_ins<=5))
            pc_sem_ins[vehicle_labels] = 1
            ped_labels = np.where((pc_sem_ins >= 6) & (pc_sem_ins <= 8))
            pc_sem_ins[ped_labels] = 2
            ground_labels = np.where((pc_sem_ins >= 9) & (pc_sem_ins <= 12))
            pc_sem_ins[ground_labels] = 3
            other_labels = np.where((pc_sem_ins > 3))
            pc_sem_ins[other_labels] = 0


        pc_sem_ins = pc_sem_ins.astype(np.int32)
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp
            #print(semins,tp)
            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

    @staticmethod
    # ==============================================================================
    #                                                         POINT_CLOUD_2_BIRDSEYE
    # ==============================================================================
    def point_cloud_2_top(points,
                          res=0.1,
                          zres=0.3,
                          side_range=(-20., 20 - 0.05),  # left-most to right-most
                          fwd_range=(0., 40. - 0.05),  # back-most to forward-most
                          height_range=(-2., 0.),  # bottom-most to upper-most
                          ):
        """ Creates an birds eye view representation of the point cloud data for MV3D.
        Args:
            points:     (numpy array)
                        N rows of points data
                        Each point should be specified by at least 3 elements x,y,z
            res:        (float)
                        Desired resolution in metres to use. Each output pixel will
                        represent an square region res x res in size.
            zres:        (float)
                        Desired resolution on Z-axis in metres to use.
            side_range: (tuple of two floats)
                        (-left, right) in metres
                        left and right limits of rectangle to look at.
            fwd_range:  (tuple of two floats)
                        (-behind, front) in metres
                        back and front limits of rectangle to look at.
            height_range: (tuple of two floats)
                        (min, max) heights (in metres) relative to the origin.
                        All height values will be clipped to this min and max value,
                        such that anything below min will be truncated to min, and
                        the same for values above max.
        Returns:
            numpy array encoding height features , density and intensity.
        """
        # EXTRACT THE POINTS FOR EACH AXIS
        x_points = points[:, 0]
        y_points = points[:, 1]
        print('xpoint range is ', min(x_points), max(x_points))
        print('ypoint range is ', min(y_points), max(y_points))
        z_points = points[:, 2]
        print('zpoint range is ', min(z_points), max(z_points))
        reflectance = points[:, 3]

        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = int((side_range[1] - side_range[0]) / res)
        y_max = int((fwd_range[1] - fwd_range[0]) / res)
        z_max = int((height_range[1] - height_range[0]) / zres)
        top = np.zeros([y_max + 1, x_max + 1, z_max + 1], dtype=np.float32)

        # FILTER - To return only indices of points within desired cube
        # Three filters for: Front-to-back, side-to-side, and height ranges
        # Note left side is positive y axis in LIDAR coordinates
        f_filt = np.logical_and(
            (x_points > fwd_range[0]), (x_points < fwd_range[1]))
        s_filt = np.logical_and(
            (y_points > -side_range[1]), (y_points < -side_range[0]))
        filt = np.logical_and(f_filt, s_filt)

        for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):
            # print('i=',i,'h=',height,'zmax=',z_max)

            z_filt = np.logical_and((z_points >= height),
                                    (z_points < height + zres))
            zfilter = np.logical_and(filt, z_filt)
            indices = np.argwhere(zfilter).flatten()

            # KEEPERS
            xi_points = x_points[indices]
            yi_points = y_points[indices]
            zi_points = z_points[indices]
            ref_i = reflectance[indices]

            # CONVERT TO PIXEL POSITION VALUES - Based on resolution
            x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
            y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

            # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
            # floor & ceil used to prevent anything being rounded to below 0 after
            # shift
            x_img -= int(np.floor(side_range[0] / res))
            y_img += int(np.floor(fwd_range[1] / res))

            # CLIP HEIGHT VALUES - to between min and max heights
            pixel_values = zi_points - height_range[0]
            # pixel_values = zi_points

            # FILL PIXEL VALUES IN IMAGE ARRAY
            top[y_img, x_img, i] = pixel_values

            # max_intensity = np.max(prs[idx])
            top[y_img, x_img, z_max] = ref_i

        top = (top / np.max(top)* 255).astype(np.uint8)
        return top

