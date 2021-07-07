# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import h5py
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from model_util_mp3d import rotate_aligned_boxes

from model_util_mp3d import MP3DDatasetConfig
DC = MP3DDatasetConfig()
#TODO
MAX_NUM_OBJ = 64
#MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# for 800 training samples
#MAX_NUM_OBJ: 28
#MAX_NUM_POINTS: 968435
#MEAN_NUM_POINTS: 193136.24625
#MEAN_RGB: [0.51826599 0.49850078 0.46754714]



class MP3DDetectionDataset(Dataset):

    def __init__(self,
                 split_set='train',
                 num_points=20000,
                 use_color=False,
                 use_height=False,
                 augment=False):

        self.root_data_path = os.path.join(BASE_DIR, 'votenet_training_data/')

        if split_set=='all':
            raise NotImplementedError
        elif split_set in ['train', 'val', 'test']:
            self.data_path = os.path.join(self.root_data_path,
                                          split_set,
                                          'votenet_inputs')
            self.files = os.listdir(self.data_path)
        else:
            raise NotImplementedError

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        file = self.files[idx]

        scan_name = file.split('.')[0]
        env_uid, ep_uid = file.split('_')

        h5file = h5py.File(os.path.join(self.data_path, file), 'r')
        mesh_vertices = np.array(h5file['point_cloud'], dtype=np.float32)
        instance_labels = np.array(h5file['instance'], dtype=np.int32)
        semantic_labels = np.array(h5file['semantic'], dtype=np.int32)
        instance_bboxes = np.array(h5file['bboxes'], dtype=np.float32)
        h5file.close()

        # convert PC to y is up.
        mesh_vertices[...,[0,1,2]] = mesh_vertices[...,[0,2,1]]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        point_cloud, choices = pc_util.random_sampling(point_cloud,
                                                       self.num_points,
                                                       return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        pcl_color = pcl_color[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            #TODO: change classe labels
            if not (semantic_labels[ind[0]] == -1):
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical

        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = instance_bboxes[:,-1]
        instance_bboxes_sids = instance_bboxes[:,-1]
        instance_bboxes_sids = instance_bboxes_sids.astype(np.int)
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[instance_bboxes_sids,:]

        #TODO: update angle_classes + residuals
        angle_residuals[0:instance_bboxes.shape[0]] = instance_bboxes[:,-2]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = instance_bboxes[:,-1]
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        #ret_dict['pcl_color'] = pcl_color
        return ret_dict


