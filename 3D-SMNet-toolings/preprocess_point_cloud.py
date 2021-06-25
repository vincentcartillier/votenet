import os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm

input_dir = 'data/point_clouds'
output_dir = 'data/preprocessed_point_clouds'

resolution = 0.03

files = [x for x in os.listdir(input_dir) if x.endswith('.ply')]


for file in files:

    point_cloud = o3d.io.read_point_cloud(os.path.join(input_dir,
                                                       file)
                                         )

    points = np.asarray(point_cloud.points)
    points[...,[0,1,2]] = points[...,[0,2,1]]
    points[...,1] *= 1
    point_cloud.points = o3d.utility.Vector3dVector(points)

    down_point_cloud = point_cloud.voxel_down_sample(voxel_size=resolution)


    o3d.io.write_point_cloud(os.path.join(output_dir,
                                          file),
                             down_point_cloud)


