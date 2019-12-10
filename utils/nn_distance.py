# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Chamfer distance in Pytorch.
Author: Charles R. Qi
"""

import tensorflow as tf
import numpy as np

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = tf.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = tf.clip_by_value(abs_error, clip_value_min=0, clip_value_max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1, 2), [1,1,M,1])
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2, 1), [1,N,1,1])
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = tf.reduce_sum(huber_loss(pc_diff, delta), -1) # (B,N,M)
    elif l1:
        pc_dist = tf.reduce_sum(tf.abs(pc_diff), -1) # (B,N,M)
    else:
        pc_dist = tf.reduce_sum(pc_diff**2, -1) # (B,N,M)
    dist1 = tf.math.reduce_min(pc_dist, 2) # (B,N)
    idx1 = tf.argmin(pc_dist, 2)
    dist2 = tf.math.reduce_min(pc_dist, 1) # (B,M)
    idx2 = tf.argmin(pc_dist, 1)
    return dist1, idx1, dist2, idx2

def demo_nn_distance():
    np.random.seed(0)
    pc1arr = np.random.random((1,5,3))
    pc2arr = np.random.random((1,6,3))
    print('-'*30)
    print('L1smooth dists:')
    pc1 = tf.convert_to_tensor(pc1arr.astype(np.float32))
    pc2 = tf.convert_to_tensor(pc2arr.astype(np.float32))
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2, True)
    with tf.Session() as sess:
        print(sess.run(dist1))
        print(sess.run(idx1))
    
if __name__ == '__main__':
    demo_nn_distance()
