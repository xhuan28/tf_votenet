import os
import sys
import tensorflow as tf
BASE_PATH = os.path.dirname((os.path.abspath(__file__)))
ROOT_PATH = os.path.dirname(BASE_PATH)
sys.path.append(os.path.join(ROOT_PATH, 'pointnet2/utils'))

from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_util

NUM_CLASS = 10
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_PROPOSAL = 256

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl

def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

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
    print('seed_xyz', l2_xyz)
    print('seed_features', l2_points)
    
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
    print('vote_xyz ', vote_xyz)
    print('vote_features', vote_features)
    
    ###########################################################################
    #  Proposal Module
    ###########################################################################
    xyz, features, indices = pointnet_sa_module(vote_xyz, vote_features, npoint=256, radius=0.3, nsample=16, mlp=[256,128,128,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='proposal_sa1')
    end_points['aggregated_vote_xyz'] = xyz
    end_points['aggregated_vote_feature'] = features
    net = tf_util.conv1d(features, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='proposal_conv1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='proposal_conv2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 2+3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4+NUM_CLASS, 1, padding='VALID', bn=False, activation_fn=None, is_training=is_training, scope='proposal_conv3', bn_decay=bn_decay)
    #end_points = decode_scores(net, end_points, NUM_CLASS, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, MEAN_SIZE_ARR)
    return net, end_points

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
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
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((1,20000,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)