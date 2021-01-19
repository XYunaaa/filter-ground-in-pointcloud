import numpy as np

def load_velo_scan(file):
    '''Load and parse a velodyne binary file'''
    file_type = file.split('.')
    if file_type[-1] == 'npy':
        scan = np.load(file)
        if scan.shape[1] ==3:
            aa = np.random.rand(scan.shape[0], 1)
            voxel_add = np.hstack((scan, aa))
        else:
            voxel_add = scan[:,:4]
    else:
        voxel_add = np.fromfile(file,dtype=np.float32)

    voxel_add = voxel_add.reshape(-1, 4)

    return voxel_add


def yield_velo_scans(velo_files):
    '''
    input:  velo_files (a file path list)
    output: Lidar point cloud iterator
    '''
    for file in velo_files:
        yield load_velo_scan(file)
