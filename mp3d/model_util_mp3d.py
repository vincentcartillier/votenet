# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

class MP3DDatasetConfig(object):
    def __init__(self):
        self.num_class = 15
        self.num_heading_bin = 1
        self.num_size_cluster = 15
        
        
        #self.num_class = 18
        #self.num_heading_bin = 1
        #self.num_size_cluster = 18


        self.type2class = {'chair': 0,
                           'sofa': 1,
                           'plant': 2,
                           'bed': 3,
                           'toilet': 4,
                           'tv_monitor': 5,
                           'shoe': 6,
                           'dietary_supp': 7,
                           'cartridge': 8,
                           'doll_toy': 9,
                           'puzzle_toy': 10,
                           'lotion': 11,
                           'vehicle_toy': 12,
                           'figurine': 13,
                           'bag': 14
                          }
        
        #self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
        #    'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
        #    'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17}  
 

        self.class2type = {self.type2class[t]:t for t in self.type2class}

        # defined from Habitat y is up
        self.type2mean_size = {
            'chair': np.array([1.008, 0.755, 0.610]),
            'sofa': np.array([2.310, 1.272, 0.855]),
            'plant': np.array([1.341, 0.793, 0.555]),
            'bed': np.array([2.260, 1.743, 0.980]),
            'toilet': np.array([0.795, 0.671, 0.503]),
            'tv_monitor': np.array([1.079, 0.722, 0.217]),

            # NOTE: Attention for new assets dims where recorded with z up.
            # has been converted to y is up here.
            'shoe': np.array([0.166, 0.19, 0.287]),
            'dietary_supp':  np.array([0.10, 0.137, 0.1]),
            'cartridge':  np.array([0.116, 0.148, 0.077]),
            'doll_toy':  np.array([0.265, 0.168, 0.235]),
            'puzzle_toy':  np.array([0.288, 0.129, 0.242]),
            'lotion':  np.array([0.11, 0.139, 0.074]),
            'vehicle_toy':  np.array([0.174, 0.126, 0.183]),
            'figurine': np.array([0.17, 0.178, 0.147]),
            'bag':  np.array([0.31, 0.279, 0.231]),

        }

        #TODO convert sizes to z is up
        for k in self.type2mean_size.keys():
            v = self.type2mean_size[k]
            self.type2mean_size[k] = np.array([v[0], v[2], v[1]])

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for k,v in self.type2class.items():
            self.mean_size_arr[v] = self.type2mean_size[k]

        # -- self.mean_size_arr = np.ones((self.num_size_cluster, 3))
        # -- # -- mp3d
        # -- self.mean_size_arr[3,:] = np.array([1.876858  , 1.84255952, 1.19315654])
        # -- self.mean_size_arr[0,:] = np.array([0.61327999, 0.61486087, 0.71827014])
        # -- self.mean_size_arr[1,:] = np.array([1.39550063, 1.51215451, 0.83443565])
        
        # -- scannet
        #self.mean_size_arr[1,:] = np.array([1.876858  , 1.84255952, 1.19315654])
        #self.mean_size_arr[2,:] = np.array([0.61327999, 0.61486087, 0.71827014])
        #self.mean_size_arr[3,:] = np.array([1.39550063, 1.51215451, 0.83443565])



        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        angle = angle%(2*np.pi)
        return 0



    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return residual

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self,
                  center,
                  heading_class,
                  heading_residual,
                  size_class,
                  size_residual):

        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        #obb[6] = heading_angle
        return obb

def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]


    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)
