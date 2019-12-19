import os
import sys
import tensorflow as tf
BASE_PATH = os.path.dirname((os.path.abspath(__file__)))
ROOT_PATH = os.path.dirname(BASE_PATH)
sys.path.append(os.path.join(ROOT_PATH, 'pointnet2/utils'))
sys.path.append(os.path.join(ROOT_PATH, 'sunrgbd'))
sys.path.append(os.path.join(ROOT_PATH, 'utils'))

from pointnet_util import pointnet_sa_module, pointnet_fp_module
from model_util_sunrgbd import SunrgbdDatasetConfig
from nn_distance import nn_distance
from nn_distance import huber_loss
import tf_util
import numpy as np

# constant for dataset
DC = SunrgbdDatasetConfig()
NUM_CLASS = 10
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_PROPOSAL = 256
MEAN_SIZE_ARR = DC.mean_size_arr
MAX_NUM_OBJ = 64
# constant for loss calculation
FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
#OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
OBJECTNESS_POS_WEIGHTS = 4

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    center_label_pl = tf.placeholder(tf.float32, shape=(batch_size, MAX_NUM_OBJ, 3))
    heading_class_label_pl = tf.placeholder(tf.int64, shape=(batch_size, MAX_NUM_OBJ))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, MAX_NUM_OBJ))
    size_class_label_pl = tf.placeholder(tf.int64, shape=(batch_size, MAX_NUM_OBJ))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, MAX_NUM_OBJ, 3))
    sem_cls_label_pl = tf.placeholder(tf.int64, shape=(batch_size, MAX_NUM_OBJ))
    box_label_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, MAX_NUM_OBJ))       
    vote_label_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    vote_label_mask_pl = tf.placeholder(tf.int64, shape=(batch_size, num_point))
    return pointclouds_pl, center_label_pl, heading_class_label_pl, heading_residual_label_pl, \
size_class_label_pl, size_residual_label_pl, sem_cls_label_pl, box_label_mask_pl, vote_label_pl, vote_label_mask_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    l0_xyz = point_cloud[..., 0:3]
    l0_points = None

    ###########################################################################
    #  PointNet2 Backbone
    ###########################################################################
    # 4 Set Abstraction Layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64, mlp=[1,64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='backbone_sa1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=32, mlp=[128,128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='backbone_sa2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=16, mlp=[256,128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='backbone_sa3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=16, mlp=[256,128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='backbone_sa4')

    # 2 Feature Propagation Layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,256,256], is_training, bn_decay, scope='backbone_fa1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,256,256], is_training, bn_decay, scope='backbone_fa2')

    end_points['seed_xyz'] = l2_xyz
    end_points['seed_features'] = l2_points
    # need to double check the following line
    end_points['seed_inds'] = l2_indices
#    print('seed_xyz', l2_xyz)
#    print('seed_features', l2_points)
    
    ###########################################################################
    #  Voting Module
    ###########################################################################
    net = tf_util.conv1d(l2_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='vote_conv1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='vote_conv2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 256+3, 1, padding='VALID', bn=False, activation_fn=None, is_training=is_training, scope='vote_conv3', bn_decay=bn_decay)
    
    vote_xyz = l2_xyz + net[:,:,0:3]
    vote_features = l2_points + net[:,:,3:]
    end_points['vote_xyz'] = vote_xyz
    end_points['vote_features'] = vote_features
#    print('vote_xyz ', vote_xyz)
#    print('vote_features', vote_features)
    
    ###########################################################################
    #  Proposal Module
    ###########################################################################
    xyz, features, indices = pointnet_sa_module(vote_xyz, vote_features, npoint=256, radius=0.3, nsample=16, mlp=[256,128,128,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='proposal_sa1')
    end_points['aggregated_vote_xyz'] = xyz
    end_points['aggregated_vote_feature'] = features
    net = tf_util.conv1d(features, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='proposal_conv1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='proposal_conv2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 2+3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4+NUM_CLASS, 1, padding='VALID', bn=False, activation_fn=None, is_training=is_training, scope='proposal_conv3', bn_decay=bn_decay)
    end_points = decode_scores(net, end_points, NUM_CLASS, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, MEAN_SIZE_ARR)
    return end_points

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    #net_transposed = tf.transpose(net, perm=[0,2,1]) # (batch_size, 1024, ..)
    net_transposed = net
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = tf.reshape(net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4], [batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(mean_size_arr.astype(np.float32)), 0), 0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

def compute_vote_loss(labels, end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = tf.to_int64(end_points['seed_inds']) # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = tf.gather(labels['vote_label_mask'], seed_inds, batch_dims=1)  #[bsize, 1024]
    #seed_inds_expand = tf.tile(tf.expand_dims(seed_inds, -1), [1,1,3*GT_VOTE_FACTOR])
    seed_gt_votes = tf.gather(labels['vote_label'], seed_inds, batch_dims=1)
    seed_gt_votes += tf.tile(end_points['seed_xyz'], [1,1,3])

    # Compute the min of min of distance
    vote_xyz_reshape = tf.reshape(vote_xyz, [batch_size*num_seed, -1, 3]) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = tf.reshape(seed_gt_votes, [batch_size*num_seed, GT_VOTE_FACTOR, 3]) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist = tf.math.reduce_min(dist2, 1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = tf.reshape(votes_dist, [batch_size, num_seed])
    vote_loss = tf.reduce_sum(votes_dist*tf.to_float(seed_gt_votes_mask))/(tf.reduce_sum(tf.to_float(seed_gt_votes_mask))+1e-6)
    return vote_loss

def compute_objectness_loss(labels, end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = labels['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = tf.sqrt(dist1+1e-6)
    objectness_mask = tf.zeros((B,K))
    objectness_label = tf.cast(euclidean_dist1<NEAR_THRESHOLD, dtype=tf.int32)
    objectness_mask = tf.bitwise.bitwise_or(tf.cast(euclidean_dist1<NEAR_THRESHOLD, dtype=tf.int32), 
                                            tf.cast(euclidean_dist1>FAR_THRESHOLD, dtype=tf.int32))

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    #criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    #objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=objectness_label, logits=objectness_scores)
    objectness_loss = tf.reduce_sum(objectness_loss * tf.to_float(objectness_mask))/(tf.to_float(tf.reduce_sum(objectness_mask))+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(labels, end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = labels['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = labels['box_label_mask']
    objectness_label = tf.to_float(end_points['objectness_label'])
    centroid_reg_loss1 = \
        tf.reduce_sum(dist1*objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        tf.reduce_sum(dist2*box_label_mask)/(tf.reduce_sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = tf.gather(labels['heading_class_label'], object_assignment, batch_dims=1) # select (B,K) from (B,K2)
#    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
#    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=heading_class_label, logits=end_points['heading_scores'])
    heading_class_loss = tf.reduce_sum(heading_class_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    heading_residual_label = tf.gather(labels['heading_residual_label'], object_assignment, batch_dims=1) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = tf.one_hot(heading_class_label, num_heading_bin) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = tf.reduce_sum(heading_residual_normalized_loss*objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = tf.gather(labels['size_class_label'], object_assignment, batch_dims=1) # select (B,K) from (B,K2)
    size_class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=size_class_label, logits=end_points['size_scores']) # (B,K)
    size_class_loss = tf.reduce_sum(size_class_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    size_residual_label = tf.gather(labels['size_residual_label'], object_assignment, batch_dims=1) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = tf.one_hot(size_class_label, num_size_cluster) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = tf.tile(tf.expand_dims(size_label_one_hot, -1), [1,1,1,3]) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = tf.reduce_sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(mean_size_arr.astype(np.float32)), 0), 0) # (1,1,num_size_cluster,3) 
    mean_size_label = tf.reduce_sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)

    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = tf.reduce_mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = tf.reduce_sum(size_residual_normalized_loss*objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = tf.gather(labels['sem_cls_label'], object_assignment, batch_dims=1) # select (B,K) from (B,K2)
    sem_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sem_cls_label, logits=end_points['sem_cls_scores'])  # (B,K)
    sem_cls_loss = tf.reduce_sum(sem_cls_loss * objectness_label)/(tf.reduce_sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(labels, end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(labels, end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(labels, end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        tf.reduce_sum(tf.to_float(objectness_label))/tf.to_float(total_num_proposal)
    end_points['neg_ratio'] = \
        tf.reduce_sum(tf.to_float(objectness_mask))/tf.to_float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(labels, end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss
    
    tf.summary.scalar('classify loss', loss)
    tf.add_to_collection('losses', loss)

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = tf.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = tf.reduce_sum(tf.to_float((obj_pred_val==tf.to_int64(objectness_label)))*tf.to_float(objectness_mask))/(tf.to_float(tf.reduce_sum(objectness_mask))+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
    
if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((1,20000,3))
        end_points = get_model(inputs, tf.constant(True), 10)
        loss, end_points = get_loss(end_points, DC)
        for key in end_points:
            print(key)
