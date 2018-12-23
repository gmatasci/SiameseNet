# TODO:
# - check tf summaries not writing proper sequence but single value

# DONE:
# - remove sigmoid of eucd as it is flipped -- done
# - get the image/CNN version to run -- issue with BN and variable reuse, had to add "with tf.variable_scope('twin_CNN', reuse=tf.AUTO_REUSE):"
# - extract meaningful probabilities from Eucl dist -- done by using binary cross entropy and sigmoid from logits
# - check why probs with binary cross entr are 0.1 for all samples -- issue with learning rate that was too high (1e-2) for
#   a too small loss function (0.60 at beginning) computed as the mean with tf.reduce_mean(), using tf.reduce_sum()
#   solves the problem but it is better to use a lower learning rate (1e-4)


""" Siamese implementation using Tensorflow with MNIST example for identification/recognition
This siamese network embeds a 28x28 image (a point in 784D)
into a lower dimensional space and then compares it to a candidate via subtraction or concatenation to see
if they represent the same number (same individual). Adapted from: https://github.com/ywpkwon/siamese_tf_mnist
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, precision_recall_fscore_support, classification_report, f1_score

import os
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from utils import *

import my_inference

PARAMS = {}
PARAMS['is_image'] = True   # is MNIST taken as images
PARAMS['comparison_type'] = 'concat' # or 'subtract'
PARAMS['patience'] = 5     # number of epochs we tolerate the validation accuracy to be stagnant (not larger than current best accuracy)
PARAMS['batch_size_trn'] = 128
PARAMS['learn_rate'] = 1e-4
PARAMS['epochs'] = 200
PARAMS['batch_size_tst'] = 500
PARAMS['load'] = False
# PARAMS['load'] = True

PARAMS['dirs'] = {}
PARAMS['dirs']['base'] = r'C:\Projects\Trials\SiameseNet'
PARAMS['dirs']['model'] = os.path.join(PARAMS['dirs']['base'], 'Models')
PARAMS['dirs']['log'] = os.path.join(PARAMS['dirs']['base'], 'Logs')
PARAMS['dirs']['fig'] = os.path.join(PARAMS['dirs']['base'], 'Figures')


def assess_classif(Y, Y_pred, normalize_conf_mat=False, verbose=False):
    """
    Asssess classification results.

    :param Y: True labels
    :param Y_pred: Predicted labels
    :return res_dict: dictionary with the results on the test set: conf_mat (true labels as rows, predicted labels as columns), OA, Kappa, class_measures
    """

    # Assess test predictions and save results
    res_dict = {}
    conf_mat = confusion_matrix(Y, Y_pred)
    if normalize_conf_mat:
        res_dict['conf_mat'] = np.round((conf_mat.astype(np.float) / conf_mat.sum(axis=1)[:, np.newaxis])*100, 1)  # normalized by true labels totals (true labels as rows, predicted labels as columns)
    else:
        res_dict['conf_mat'] = conf_mat
    res_dict['OA'] = np.round(accuracy_score(Y, Y_pred)*100, 2)
    res_dict['Kappa'] = cohen_kappa_score(Y, Y_pred)
    res_dict['Mean_F1_score'] = f1_score(Y, Y_pred, average='macro')  # averaged individual class F1-scores. With average='macro': unweighted average, with average='weighted': weights proportional to support (the number of true instances for each label)
    res_dict['precision'], res_dict['recall'], res_dict['f1'], _ = precision_recall_fscore_support(Y, Y_pred)
    res_dict['classif_report'] = classification_report(Y, Y_pred)

    if verbose:
        print('Classification results:\n\n '
              'Confusion matrix:\n %s \n\n '
              'OA=%.2f, Kappa=%.3f, Mean F1 score=%.3f \n\n '
              'Class-specific measures:\n %s'
              % (res_dict['conf_mat'], res_dict['OA'], res_dict['Kappa'], res_dict['Mean_F1_score'], res_dict['classif_report']))

    return res_dict



# Prepare data and tf.session
mnist = input_data.read_data_sets(os.path.join(PARAMS['dirs']['base'], 'MNIST_data'), one_hot=False)

nr_val_samples = len(mnist.validation.labels)
val1_idx = np.arange(int(np.floor(nr_val_samples / 2)))
val2_idx = np.arange(int(np.floor(nr_val_samples/2)), nr_val_samples)
val_y1 = mnist.validation.labels[val1_idx]
val_y2 = mnist.validation.labels[val2_idx]
val_y = (val_y1 == val_y2).astype('float')

if PARAMS['is_image']:
    val_set = np.reshape(mnist.validation.images, (-1, 28, 28, 1))
else:
    val_set = mnist.validation.images
val_x1 = val_set[val1_idx]
val_x2 = val_set[val2_idx]

sess = tf.InteractiveSession()

# Setup siamese network
siamese = my_inference.SiameseNet(is_image=PARAMS['is_image'], comparison_type=PARAMS['comparison_type'])
# train_step = tf.train.GradientDescentOptimizer(learning_rate=PARAMS['learn_rate']).minimize(siamese.loss)
train_step = tf.train.AdamOptimizer(learning_rate=PARAMS['learn_rate']).minimize(siamese.loss)

saver = tf.train.Saver()
tf.global_variables_initializer().run()

writer = tf.summary.FileWriter(PARAMS['dirs']['log'], sess.graph)

# Start training
if PARAMS['load']: saver.restore(sess, PARAMS['dirs']['model'])

# Maximum number of steps/epoch, time 2 as we use 2*batch_size samples each time
batch_steps = np.ceil(mnist.train.num_examples / (2*PARAMS['batch_size_trn'])).astype(np.int32)

# Training cycle
best_loss_val = np.inf
best_epoch = 0
loss_val_list = []  # store trn OA and loss for averaging over the epoch
val_OA_list = []
print("Start training...")
for epoch in range(PARAMS['epochs']):
    start_epoch = tic()
    loss_train_list = []  # store trn OA and loss for averaging over the epoch
    for step in range(batch_steps):

        batch_x1, batch_y1 = mnist.train.next_batch(PARAMS['batch_size_trn'])
        batch_x2, batch_y2 = mnist.train.next_batch(PARAMS['batch_size_trn'])
        if PARAMS['is_image']:
            batch_x1 = np.reshape(batch_x1, (PARAMS['batch_size_trn'], 28, 28, 1))
            batch_x2 = np.reshape(batch_x2, (PARAMS['batch_size_trn'], 28, 28, 1))

        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_train = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y,
                            siamese.is_training: True
        })

        if np.isnan(loss_train):
            print('Model diverged with loss = NaN')
            quit()

        # if step % 100 == 0:
        #     print ('Step %d: training loss %.3f' % (step, loss_train))

        loss_train_list.append(loss_train)  # grow lists

        ## Assess model at the end of the epoch
        if step == batch_steps - 1:

            pair_prob, loss_val = sess.run([siamese.pair_prob, siamese.loss], feed_dict={
                siamese.x1: val_x1,
                siamese.x2: val_x2,
                siamese.y_: val_y,
                siamese.is_training: False
            })

            if isinstance(loss_val, list):
                loss_val = loss_val[0]
            loss_val_list.append(loss_val)  # grow lists
            loss_val_summary = tf.summary.scalar('validation_loss', loss_val)
            writer.add_summary(loss_val_summary.eval(), epoch)

            ## Update best val OA only if current one exceeds previous best value
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                best_epoch = epoch
                epochs_no_improv = 0  # reset value to compare with patience
                best_model_dir = os.path.join(PARAMS['dirs']['model'], 'best_model')  # save model with checkpoint in specific hyperpar folder
                saver.save(sess, best_model_dir)
            else:
                epochs_no_improv += 1

            y_pred = (pair_prob > 0.5).astype(np.int)
            res_dict = assess_classif(val_y, y_pred, normalize_conf_mat=False, verbose=False)
            val_OA = res_dict['OA']
            val_F1 = res_dict['f1'][1]  # of the positive class, the matches
            val_OA_list.append(val_OA)

    # Compute epoch means
    mean_loss_train = np.mean(loss_train_list)
    print("Epoch %g: train loss = %.4f, val loss = %.4f, val OA = %.2f, val F1 = %.3f, %s" % (
    epoch, mean_loss_train, loss_val, val_OA, val_F1, toc(start_epoch)))

    ## Earlystopping with a given patience value
    if epochs_no_improv > PARAMS['patience']:
        print("Patience reached: earlystopping with best val loss = %g (epoch %i)" % (best_loss_val, best_epoch))
        break

plt.figure()
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('val loss')
plt.savefig(os.path.join(PARAMS['dirs']['fig'], '%s.%s' % ('val_loss', 'png')), dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(val_OA_list)
plt.xlabel('epochs')
plt.ylabel('val OA')
plt.savefig(os.path.join(PARAMS['dirs']['fig'], '%s.%s' % ('val_OA', 'png')), dpi=400, bbox_inches='tight')
plt.close()

writer.close()

## Prediction on test set

best_model_dir = os.path.join(PARAMS['dirs']['model'], 'best_model')  # points to folder with best model for this combination of hyperparameters
saver.restore(sess, best_model_dir)

batch_steps = np.ceil(mnist.test.num_examples / (2*PARAMS['batch_size_tst'])).astype(np.int32)
pair_prob_list = []
tst_y = []
for step in range(batch_steps):

    batch_x1, batch_y1 = mnist.test.next_batch(PARAMS['batch_size_tst'])
    batch_x2, batch_y2 = mnist.test.next_batch(PARAMS['batch_size_tst'])
    if PARAMS['is_image']:
        batch_x1 = np.reshape(batch_x1, (PARAMS['batch_size_tst'], 28, 28, 1))
        batch_x2 = np.reshape(batch_x2, (PARAMS['batch_size_tst'], 28, 28, 1))

    batch_y = (batch_y1 == batch_y2).astype('float')
    tst_y.extend(batch_y)

    pair_prob = sess.run([siamese.pair_prob], feed_dict={
        siamese.x1: batch_x1,
        siamese.x2: batch_x2,
        siamese.is_training: False
    })

    if isinstance(pair_prob, list):
        pair_prob = pair_prob[0]
    pair_prob_list.extend(pair_prob)

    if not PARAMS['is_image']:
        batch_x1 = np.reshape(batch_x1, (-1, 28, 28, 1))
        batch_x2 = np.reshape(batch_x2, (-1, 28, 28, 1))

    nr_plots = 1
    nr_pairs_per_plot = 10
    for i_plot in range(nr_plots):
        offset = np.random.randint(0,len(batch_y)-nr_pairs_per_plot)
        fig, ax = plt.subplots(nrows=2, ncols=nr_pairs_per_plot)
        for i in range(nr_pairs_per_plot):
            ax[0, i].set_title('%i: %.2f' % (int(batch_y[offset+i]), pair_prob[offset+i]), fontsize=6)
            ax[0, i].imshow(np.squeeze(batch_x1[offset+i,:,:,:]))
            ax[0, i].axis('off')
            ax[1, i].imshow(np.squeeze(batch_x2[offset+i,:,:,:]))
            ax[1, i].axis('off')
        plt.savefig(os.path.join(PARAMS['dirs']['fig'], 'sanity_check_step%i_plot%i.%s' % (step, i_plot, 'png')), dpi=400, bbox_inches='tight')
        plt.close()

y_pred = (np.array(pair_prob_list) > 0.5).astype(np.int)
RES_tst = assess_classif(tst_y, y_pred, normalize_conf_mat=False, verbose=True)

