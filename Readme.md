# Filter ground in pointclouds 
Five filter ground methods；
## Example
![image](https://github.com/XYunaaa/filter-ground-in-pointcloud/blob/master/demo.png)


## Evalution

We merge five different ground labels in SemanticKITTI (i.e., road, parking, side walk, other ground and terrain) as
ground truth. Table II shows the results of the five different ground filters adopted on the total 11 sequences(00-10) of
segmentation data. Considering we only focus on filtering
the ground points, the Ground IOU (intersection-over-union
of the ground point cloud) is used to
evaluate five mentioned filters. The larger the Ground IOU,
the higher the accuracy of ground detection. At the same time, filters can also cause false detections,
so we also analyzed the detection errors on the three detection
targets, e.g., the CarErr, PedErr and CycErr(The error of each
class) .The lower the Error,
the higher the accuracy of ground detection.

## Install
    main package : open3d-0.9.0 numpy 

    ground_removal_ext https://github.com/HViktorTsoi/pointcloud_ground_removal
## Usage
    
    python filter_ground.py ; 
    # Change filter ground method by modifying methodnum (in 0-4) in the filterGround function;
