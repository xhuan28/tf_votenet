'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))

import votenet
import sunrgbd_detection_dataset
from model_util_sunrgbd import SunrgbdDatasetConfig

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL_FILE = os.path.join(ROOT_DIR, 'models', 'votenet.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 10

TRAIN_DATASET = sunrgbd_detection_dataset.SunrgbdDetectionVotesDataset(split_set='train', batch_size=BATCH_SIZE, num_points=NUM_POINT, augment=True)
TEST_DATASET = sunrgbd_detection_dataset.SunrgbdDetectionVotesDataset(split_set='val', batch_size=BATCH_SIZE, num_points=NUM_POINT, augment=False)
DC = SunrgbdDatasetConfig()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, center_label_pl, heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl, sem_cls_label_pl, box_label_mask_pl, \
            vote_label_pl, vote_label_mask_pl = votenet.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            labels_pl = {}
            labels_pl['center_label'] = center_label_pl
            labels_pl['heading_class_label'] = heading_class_label_pl
            labels_pl['heading_residual_label'] = heading_residual_label_pl
            labels_pl['size_class_label'] = size_class_label_pl
            labels_pl['size_residual_label'] = size_residual_label_pl
            labels_pl['sem_cls_label'] = sem_cls_label_pl
            labels_pl['box_label_mask'] = box_label_mask_pl
            labels_pl['vote_label'] = vote_label_pl
            labels_pl['vote_label_mask'] = vote_label_mask_pl
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            end_points = votenet.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            
            loss, end_points = votenet.get_loss(labels_pl, end_points, DC)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

#            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
#            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
#            tf.summary.scalar('accuracy', accuracy)

            print('Get training operator')
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ops = {'pointclouds_pl': pointclouds_pl,
               'center_label_pl': center_label_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'sem_cls_label_pl': sem_cls_label_pl,
               'box_label_mask_pl': box_label_mask_pl,
               'vote_label_pl': vote_label_pl,
               'vote_label_mask_pl': vote_label_mask_pl,
               'is_training_pl': is_training_pl,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

#        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data_label = TRAIN_DATASET.next_batch()
        
        feed_dict = {ops['pointclouds_pl']: batch_data_label['point_clouds'],
                     ops['center_label_pl']: batch_data_label['center_label'],
                     ops['heading_class_label_pl']: batch_data_label['heading_class_label'],
                     ops['heading_residual_label_pl']: batch_data_label['heading_residual_label'],
                     ops['size_class_label_pl']: batch_data_label['size_class_label'],
                     ops['size_residual_label_pl']: batch_data_label['size_residual_label'],
                     ops['sem_cls_label_pl']: batch_data_label['sem_cls_label'],
                     ops['box_label_mask_pl']: batch_data_label['box_label_mask'],
                     ops['vote_label_pl']: batch_data_label['vote_label'],
                     ops['vote_label_mask_pl']: batch_data_label['vote_label_mask'],
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val = sess.run(
                [ops['merged'],
                 ops['step'],
                 ops['train_op'], 
                 ops['loss'],], feed_dict=feed_dict)
        print('*****')
        print(loss_val)
        print('*****')
        train_writer.add_summary(summary, step)
#        pred_val = np.argmax(pred_val, 1)
#        correct = np.sum(pred_val[0:bsize] == batch_data_label[0:bsize])
#        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0

    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        batch_data_label = TRAIN_DATASET.next_batch()

        feed_dict = {ops['pointclouds_pl']: batch_data_label['point_clouds'],
                     ops['center_label_pl']: batch_data_label['center_label'],
                     ops['heading_class_label_pl']: batch_data_label['heading_class_label'],
                     ops['heading_residual_label_pl']: batch_data_label['heading_residual_label'],
                     ops['size_class_label_pl']: batch_data_label['size_class_label'],
                     ops['size_residual_label_pl']: batch_data_label['size_residual_label'],
                     ops['sem_cls_label_pl']: batch_data_label['sem_cls_label'],
                     ops['box_label_mask_pl']: batch_data_label['box_label_mask'],
                     ops['vote_label_pl']: batch_data_label['vote_label'],
                     ops['vote_label_mask_pl']: batch_data_label['vote_label_mask'],
                     ops['is_training_pl']: is_training,}

        summary, step, loss_val = sess.run([ops['merged'], ops['step'],
            ops['loss']], feed_dict=feed_dict)
    
        test_writer.add_summary(summary, step)
        #pred_val = np.argmax(pred_val, 1)
        #correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        #total_correct += correct
        #total_seen += bsize
        #loss_sum += loss_val
        #batch_idx += 1
#        for i in range(0, bsize):
#            l = batch_label[i]
#            total_seen_class[l] += 1
#            total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    #return total_correct/float(total_seen)
    return 1


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
