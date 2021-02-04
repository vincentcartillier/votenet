# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import json
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, '../models'))
from pc_util import random_sampling, read_ply
from pc_util import write_oriented_bbox
from pc_util import write_oriented_bbox_color
from ap_helper import parse_predictions


sample_id = 'stage_0.scene_apt_0.id_0'
pc_path = 'data/point_clouds/pc_{}.json'.format(sample_id)
output_dir = 'data/votenet_results_scannet/{}'.format(sample_id)

label_colours = [(106, 137, 204, 255),
                 (230, 126, 34, 255), 
                 (7, 153, 146, 255),  
                 (248, 194, 145, 255),
                 (76, 209, 55, 255),  
                 (255, 168, 1, 255),  
                 (184, 233, 148, 255),
                 (39, 174, 96, 255),  
                 (229, 80, 57, 255),  
                 (30, 55, 153, 255),  
                 (24, 220, 255, 255), 
                 (234, 32, 39, 255),
                 (23, 197, 62, 255), 
                 (141, 168, 145, 255), 
                 (142, 151, 136, 255),
                 (115, 201, 77, 255), 
                 (100, 216, 255, 255), 
                 (57, 156, 36, 255),
                 (88, 108, 129, 255), 
                 (105, 129, 112, 255), 
                 (42, 137, 126, 255),
                 (155, 108, 249, 255), 
                 (166, 148, 143, 255), 
                 (81, 91, 87, 255),
                 (100, 124, 51, 255), 
                 (73, 131, 121, 255), 
                 (157, 210, 220, 255),
                 (134, 181, 60, 255), 
                 (221, 223, 147, 255), 
                 (123, 108, 131, 255),
                 (161, 66, 179, 255), 
                 (163, 221, 160, 255), 
                 (31, 146, 98, 255),
                 (99, 121, 30, 255), 
                 (49, 89, 240, 255), 
                 (116, 108, 9, 255),
                 (161, 176, 169, 255), 
                 (80, 29, 135, 255), 
                 (177, 105, 197, 255),
                ]

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    #point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__=='__main__':
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, '../sunrgbd'))
        from sunrgbd_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        #pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, '../scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        #pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    else:
        print('Unkown dataset %s. Exiting.'%(DATASET))
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    #point_cloud = read_ply(pc_path)
    #TODO: load preprocessed point cloud
    point_cloud = json.load(open(pc_path, 'r'))
    point_cloud = np.array(point_cloud)
    #point_cloud[...,[0,1,2]] = point_cloud[...,[0,2,1]]
    #point_cloud[...,1] *= 1
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
  
    #dump_dir = os.path.join(output_dir, '%s_results'%(FLAGS.dataset))
    #dump_dir = os.path.join(output_dir, '3d_detection_results')
    #if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    #MODEL.dump_results(end_points, dump_dir, DC, True)
    #print('Dumped detection results to folder %s'%(dump_dir))
    
    #Save object-based map
    from dump_helper import softmax
    if not os.path.exists(output_dir): os.mkdir(output_dir) 
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3
    point_clouds = end_points['point_clouds'].cpu().numpy()
    sem_cls_probs = softmax(end_points['sem_cls_scores'].detach().cpu().numpy()) # B,num_proposal,10
    pred_sem_cls = np.argmax(sem_cls_probs,-1) # B,num_proposal
    pred_sem_cls_prob = np.max(sem_cls_probs,-1) # B,num_proposal
    batch_size = point_clouds.shape[0]
    pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0
    DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.
    for i in range(batch_size):
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)
        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            semantics = []
            colors = []
            for j in range(num_proposal):
                obb = DC.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                pred_size_class[i,j], pred_size_residual[i,j])
                obbs.append(obb)
                semantics.append(pred_sem_cls[i,j])
                colors.append(label_colours[pred_sem_cls[i,j]])
            colors = np.array(colors)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                #write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:],os.path.join(output_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                #write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH,pred_mask[i,:]==1),:],os.path.join(output_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                #write_oriented_bbox(obbs[pred_mask[i,:]==1,:],os.path.join(output_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                #write_oriented_bbox(obbs, os.path.join(output_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))

                write_oriented_bbox_color(obbs[objectness_prob>DUMP_CONF_THRESH,:],
                                          colors[objectness_prob>DUMP_CONF_THRESH,:],
                                          os.path.join(output_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                write_oriented_bbox_color(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH,pred_mask[i,:]==1),:],
                                          colors[np.logical_and(objectness_prob>DUMP_CONF_THRESH,pred_mask[i,:]==1),:],
                                          os.path.join(output_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                write_oriented_bbox_color(obbs[pred_mask[i,:]==1,:],
                                          colors[pred_mask[i,:]==1,:],
                                          os.path.join(output_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                write_oriented_bbox_color(obbs, 
                                          colors, 
                                          os.path.join(output_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))


