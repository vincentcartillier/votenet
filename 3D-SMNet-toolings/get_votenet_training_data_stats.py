import os
import h5py
import numpy as np
from tqdm import tqdm

data_path = 'mp3d/votenet_training_data/train/votenet_inputs/'

num_points = []
num_objects = []
rgb_values = []
for file in tqdm(os.listdir(data_path)):
    h5file = h5py.File(os.path.join(data_path, file), 'r')
    pc = np.array(h5file['point_cloud'])
    boxes = np.array(h5file['bboxes'])
    h5file.close()
    num_points.append(len(pc))
    num_objects.append(len(boxes))
    rgb_values+= pc[:,3:6].tolist()

rgb_values = np.array(rgb_values)
rgb_avg = np.mean(rgb_values, axis=0)
MAX_NUM_OBJ = max(num_objects)
MAX_NUM_POINTS = max(num_points)
MEAN_NUM_POINTS = np.mean(num_points)

print(f'MAX_NUM_OBJ: {MAX_NUM_OBJ}')
print(f'MAX_NUM_POINTS: {MAX_NUM_POINTS}')
print(f'MEAN_NUM_POINTS: {MEAN_NUM_POINTS}')
print(f'MEAN_RGB: {rgb_avg}')


