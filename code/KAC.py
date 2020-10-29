# coding=utf-8
'''
Consider different initialization
'''


from __future__ import absolute_import
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import tensorflow as tf
import numpy as np
import logging
from time import time, strftime, localtime, sleep
from KAC_Dataset import Dataset, FeatureEmbedding, UserTagEmbedding
import KAC_BatchGen as batchGenerate
import KAC_Evaluate as evaluate

import argparse
import json
from gensim import models, corpora, similarities


def parse_args():
    parser = argparse.ArgumentParser(description="Run KAC.")
    parser.add_argument('--path', nargs='?', default='Data/douban/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='douban',
                        help='Choose a dataset.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs.')
    parser.add_argument('--weight_size', type=int, default=16,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7,1e-5]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=0,
                        help='Calculate training loss or not')
    parser.add_argument('--beta', type=float, default=1,
                        help='Index of coefficient of sum of exp(A)')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--activation', type=int, default=0,
                        help='Activation for ReLU, sigmoid, tanh.')
    parser.add_argument('--algorithm', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--topk', nargs='?', default='1,3,5',
                        help='Recommend top K items.')
    parser.add_argument('--embed_type', nargs='?', default='feature',
                        help='Initialize embedding method: random, or feature.')
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='Sparsity of the dataset')
    parser.add_argument('--pos_neg', nargs='?', default='5:95',
                        help='Number oF Positive and negative examples for test.')
    parser.add_argument('--prediction', type=int, default=0,
                        help='0 for mlp-prediction_prod, 1 for mlp-prediction_concat, 2 for matmul')
    parser.add_argument('--prediction_layer', nargs='?', default='[64,32,16]',
                        help='Neurons in each MLP layer for prediction.')

    parser.add_argument('--feature_merge', nargs='?', default='conv',
                        help='The way to merge feature embedding vectors.')
    parser.add_argument('--user_gamma', type=float, default=0,
                        help='Ratio how much user personal information attribute embedding add to user historical item embedding')
    parser.add_argument('--memory', type=int, default=0,
                        help='memory length, 0 for all history')
    parser.add_argument('--attention', type=int, default=1,
                        help='flag for attention. 1: use attention; 0: use average')
    return parser.parse_args()


class KAC:
    def __init__(self, num_items, args):
        global istraining
        istraining = True
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.weight_size = args.weight_size
        self.prediction_layer = eval(args.prediction_layer)

        self.alpha = args.alpha
        self.beta = args.beta
        self.verbose = args.verbose
        self.activation = args.activation
        self.algorithm = args.algorithm
        self.batch_size = args.batch_size
        self.lambda_bilinear, self.gamma_bilinear, self.eta_bilinear = eval(args.regs)
        self.train_loss = args.train_loss

        self.algorithm_pred = args.prediction
        self.embed_type = args.embed_type
        self.embedding_kg = []
        self.embedding_tag_item = []
        self.embedding_feature_item_train = []
        self.embedding_feature_matrix = []
        self.embedding_feature_index = []
        self.feature_dims = []
        self.feature_num = []

        self.user_attribute_ratio = args.user_gamma
        self.user_attribute_num = 0
        self.user_attribute_dim = 20
        self.embedding_usertag_index = []
        self.feature_merge = args.feature_merge if self.embed_type == 'feature' else ''
        self.model_with_attention = args.attention
        self.A = []

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_id = tf.placeholder(tf.int32, shape=[None, 1])  # the id of users
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the items rated of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])  # the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # the ground truth

    def _create_variables(self):
        trainable_flag = True
        self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
        if self.embed_type == 'random':
            print('Random initialization')
            with tf.name_scope("embedding"):
                self.c1 = tf.Variable(
                    tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1',
                    dtype=tf.float32)
                self.embedding_Q_ = tf.concat([self.c2, self.c1], 0, name='embedding_Q_')
                self.embedding_Q = tf.Variable(
                    tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                    name='embedding_Q', dtype=tf.float32, trainable=True)

        if self.embed_type == 'feature':
            print('Initialization with all item features')
            with tf.name_scope("feature_embedding"):
                embed_matrix_dim = self.feature_dims[-1]
                self.embedding_Q_ = tf.Variable(tf.truncated_normal(shape=[embed_matrix_dim, self.embedding_size],
                                                                    mean=0.0, stddev=0.1),
                                                name='embedding_Q_', dtype=tf.float32, trainable=True)
                self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[embed_matrix_dim, self.embedding_size],
                                                                   mean=0.0, stddev=0.1),
                                               name='embedding_Q', dtype=tf.float32, trainable=True)

        self.bias = tf.Variable(tf.zeros(self.num_items), name='bias', trainable=trainable_flag)
        # Variables for attention
        r = (self.algorithm + 1) * self.embedding_size
        if self.feature_merge == 'concat':
            r *= sum(self.feature_num)

        self.W = tf.Variable(tf.truncated_normal(shape=[r, self.weight_size], mean=0.0,
                                                 stddev=tf.sqrt(tf.div(2.0, self.weight_size + r))),
                             name='Weights_for_MLP', dtype=tf.float32, trainable=True)

        self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(
            tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP', dtype=tf.float32,
                             trainable=True)
        self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)
        # embedding matrix for user tags
        self.user_tag_coeff = tf.Variable(1.0, name='user_tag_coeff', dtype=tf.float32, trainable=True)
        self.u = tf.Variable(tf.truncated_normal(shape=[self.user_attribute_num, self.embedding_size],
                                                 mean=0.0, stddev=0.01),
                             name='u', dtype=tf.float32, trainable=True)
        self.embedding_U = tf.concat([self.c2, self.u], 0, name='embedding_U')

    def _attention_MLP(self, q_):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]  # batch size
            n = tf.shape(q_)[1]  # user history length
            r = (self.algorithm + 1) * self.embedding_size
            if self.feature_merge == 'concat':
                r *= sum(self.feature_num)

            MLP_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b
            if self.activation == 0:
                MLP_output = tf.nn.relu(MLP_output)
            elif self.activation == 1:
                MLP_output = tf.nn.sigmoid(MLP_output)
            elif self.activation == 2:
                MLP_output = tf.nn.tanh(MLP_output)

            A_ = tf.reshape(tf.matmul(MLP_output, self.h), [b, n])

            # softmax for not mask features
            exp_A_ = tf.exp(A_)  # (b, n)
            num_idx = tf.reduce_sum(self.num_idx, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen=n, dtype=tf.float32)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            self.A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(self.A * self.embedding_q_, 1)  # (b, n, 1) * (b, n, e) -> (b, e)

    def _feature_lookup(self, input_data, query):
        b = tf.shape(input_data)[0]
        n = tf.shape(input_data)[1]  # n for user, 1 for item
        item_index2feature = tf.nn.embedding_lookup(self.embedding_feature_index, input_data)  # (b, n, f)

        if query == 'user':
            item_feature_embedding = tf.nn.embedding_lookup(
                self.embedding_Q_, item_index2feature)  # (b, n, f, e)
        else:  # query == 'item'
            item_feature_embedding = tf.nn.embedding_lookup(
                self.embedding_Q, item_index2feature)  # (b, 1, f, e)

        if self.feature_merge == 'conv':
            item_embed = self._item_feature_conv_process(item_feature_embedding, query)
        if self.feature_merge == 'avg':  # average pooling
            item_embed = tf.reduce_mean(item_feature_embedding, axis=2)
        if self.feature_merge == 'concat':
            _last_dim = sum(self.feature_num) * self.embedding_size
            item_embed = tf.reshape(item_feature_embedding, [b, n, _last_dim])

        return item_embed


    def _item_feature_conv_process(self, input_tensor, query):
        '''
        input dim: [batch_size, n, feature_num, embed_size, channel=1]
        filter dim:[filter_depth, filter_height, filter_width, in_channels, out_channels]
        '''
        # add the channel dim
        input_tensor = tf.expand_dims(input_tensor, axis=-1)

        if query == 'user':
            with tf.variable_scope('convolution_user', reuse=tf.AUTO_REUSE):
                item_feature_filter = tf.get_variable(name='single_filter_user',
                                                      shape=[1, sum(self.feature_num), 1, 1, 1], dtype=tf.float32,
                                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
                item_feature_bias = tf.get_variable(name='cnn_bias_user',
                                                    shape=[1], dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer(stddev=0.001))
                item_feature_conv = tf.nn.conv3d(input=input_tensor, filter=item_feature_filter,
                                                 strides=[1, 1, 1, 1, 1], padding='VALID')
        elif query == 'item':
            with tf.variable_scope('convolution_item', reuse=tf.AUTO_REUSE):
                item_feature_filter = tf.get_variable(name='single_filter_item',
                                                      shape=[1, sum(self.feature_num), 1, 1, 1], dtype=tf.float32,
                                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
                item_feature_bias = tf.get_variable(name='cnn_bias_item',
                                                    shape=[1], dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer(stddev=0.001))
                item_feature_conv = tf.nn.conv3d(input=input_tensor, filter=item_feature_filter,
                                                 strides=[1, 1, 1, 1, 1], padding='VALID')
        else:
            raise exception('Input type error! Neither user or item.')

        output = tf.layers.batch_normalization(item_feature_conv + item_feature_bias, training=istraining)
        output = tf.squeeze(output)
        if query == 'item':
            output = tf.expand_dims(output, axis=1)
        return output

    def _user_attribute_conv_process(self, input_tensor):
        '''
        input dim: [batch_size, 1, user_feature_num, embed_size]
        filter dim:[filter_height, filter_width, in_channels, out_channels]
        '''
        input_tensor = tf.expand_dims(input_tensor, axis=-1)
        with tf.variable_scope('user_attribute_convolution', reuse=tf.AUTO_REUSE):
            user_attribute_filter = tf.get_variable(name='user_attr_single_filter',
                                                    shape=[1, self.user_attribute_dim, 1, 1, 1], dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer(stddev=0.01))
            user_attribute_bias = tf.get_variable(name='user_attr_cnn_bias',
                                                  shape=[1], dtype=tf.float32,
                                                  initializer=tf.truncated_normal_initializer(stddev=0.001))
            user_attribute_conv = tf.nn.conv3d(input=input_tensor, filter=user_attribute_filter,
                                               strides=[1, 1, 1, 1, 1], padding='VALID')

        output = tf.layers.batch_normalization(user_attribute_conv + user_attribute_bias, training=istraining)
        output = tf.squeeze(output)

        return output  # (b, e)


    def _create_inference(self):
        with tf.name_scope("inference"):
            if self.embed_type == 'feature':
                print('feature embedding...')
                self.embedding_q_ = self._feature_lookup(self.user_input, 'user')  # (b, n, f*e)
                self.embedding_q = self._feature_lookup(self.item_input, 'item')  # (b, 1, f*e)

            else:
                self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input)  # (b, n, e)
                self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.model_with_attention:
                if self.algorithm == 0:  # element-wise product
                    self.embedding_p = self._attention_MLP(self.embedding_q_ * self.embedding_q)  # (b, e)
                else:  # concat
                    n = tf.shape(self.user_input)[1]
                    self.embedding_p = self._attention_MLP(
                        tf.concat([self.embedding_q_, tf.tile(self.embedding_q, tf.stack([1, n, 1]))], 2))
            else:
                self.embedding_p = tf.reduce_mean(self.embedding_q_, 1)

            self.embedding_q = tf.reduce_sum(self.embedding_q, 1)  # (b, 1, e) -> (b, e)

            if self.user_attribute_ratio != 0:
                # consider user tags
                usertag_index2feature = tf.nn.embedding_lookup(self.embedding_usertag_index, self.user_id)  # (b, 1, k)
                usertag_embedding = tf.nn.embedding_lookup(self.embedding_U, usertag_index2feature)  # (b, 1, k, e)
                self.embedding_p_info = self._user_attribute_conv_process(usertag_embedding)

                alpha = self.user_tag_coeff * self.user_attribute_ratio if self.user_attribute_ratio > 0 \
                    else self.user_tag_coeff / tf.exp(tf.to_float(self.num_idx))

                self.embedding_p = self.embedding_p * alpha + self.embedding_p_info * (1-alpha)

            bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            if self.algorithm_pred == 2:
                coeff = tf.pow(self.num_idx, tf.constant(self.alpha, tf.float32, [1]))
                self.output = tf.sigmoid(
                    coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p * self.embedding_q, 1), 1) + bias_i)
            else:
                mlp_output = self._mlp_prediction(num_layer=self.prediction_layer)
                self.output = tf.sigmoid(mlp_output)


    def _mlp_prediction(self, num_layer):
        r = sum(self.feature_num) if self.feature_merge == 'concat' else 1

        if self.algorithm_pred == 0:  # element-wise product
            input_mlp = self.embedding_p * self.embedding_q  # (b, r*e) * (b, r*e)
        else:
            input_mlp = tf.concat([self.embedding_p, self.embedding_q], axis=1)  # (b, r*2e)

        coefficient = r * (self.algorithm_pred + 1) * self.embedding_size

        i = 0
        self.W_pred_0 = tf.Variable(tf.truncated_normal(shape=[coefficient, num_layer[i]], mean=0.0,
                                                        stddev=tf.sqrt(tf.div(2.0, num_layer[i] + coefficient))),
                                    name='Weights_%s_for_MLP' % i, dtype=tf.float32, trainable=True)
        self.b_pred_0 = tf.Variable(tf.truncated_normal(shape=[1, num_layer[i]], mean=0.0, stddev=tf.sqrt(
            tf.div(2.0, num_layer[i] + self.embedding_size))), name='Bias_%s_for_MLP' % i, dtype=tf.float32,
                             trainable=True)
        layer_value = tf.matmul(input_mlp, self.W_pred_0) + self.b_pred_0
        MLP_output = tf.nn.relu(layer_value)

        i = 1
        self.W_pred_1 = tf.Variable(tf.truncated_normal(shape=[num_layer[i - 1], num_layer[i]], mean=0.0,
                                                        stddev=tf.sqrt(tf.div(2.0, num_layer[i - 1] + num_layer[i]))),
                                    name='Weights_%s_for_MLP' % i, dtype=tf.float32, trainable=True)
        self.b_pred_1 = tf.Variable(tf.truncated_normal(shape=[1, num_layer[i]], mean=0.0, stddev=tf.sqrt(
            tf.div(2.0, num_layer[i] + self.embedding_size))), name='Bias_%s_for_MLP' % i, dtype=tf.float32,
                             trainable=True)

        layer_value = tf.matmul(MLP_output, self.W_pred_1) + self.b_pred_1
        MLP_output = tf.nn.relu(layer_value)

        i = 2
        self.W_pred_2 = tf.Variable(tf.truncated_normal(shape=[num_layer[i - 1], num_layer[i]], mean=0.0,
                                                        stddev=tf.sqrt(tf.div(2.0, num_layer[i - 1] + num_layer[i]))),
                                    name='Weights_%s_for_MLP' % i, dtype=tf.float32, trainable=True)
        self.b_pred_2 = tf.Variable(tf.truncated_normal(shape=[1, num_layer[i]], mean=0.0, stddev=tf.sqrt(
            tf.div(2.0, num_layer[i] + self.embedding_size))), name='Bias_%s_for_MLP' % i, dtype=tf.float32,
                                    trainable=True)

        layer_value = tf.matmul(MLP_output, self.W_pred_2) + self.b_pred_2
        MLP_output = tf.nn.relu(layer_value)

        self.W_pred_out = tf.Variable(tf.truncated_normal(shape=[num_layer[-1], 1], mean=0.0,
                                                          stddev=tf.sqrt(tf.div(2.0, num_layer[-1] + 1))),
                                      name='Weights_-1_for_MLP', dtype=tf.float32, trainable=True)

        self.b_pred_out = tf.Variable(tf.sqrt(tf.div(2.0, num_layer[-1] + self.embedding_size)),
                                      name='Bias_-1_for_MLP')

        MLP_output = tf.matmul(MLP_output, self.W_pred_out) + self.b_pred_out

        return MLP_output

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                        self.eta_bilinear * tf.reduce_sum(tf.square(self.W))
            if self.user_attribute_ratio:
                self.loss += self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_U))
            if self.algorithm_pred != 2:
                self.loss += self.eta_bilinear * tf.reduce_sum(tf.square(self.W_pred_0)) + \
                             self.eta_bilinear * tf.reduce_sum(tf.square(self.W_pred_1)) + \
                             self.eta_bilinear * tf.reduce_sum(tf.square(self.W_pred_2)) + \
                             self.eta_bilinear * tf.reduce_sum(tf.square(self.W_pred_out))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)

    def _load_user_info(self, path):
        User_tag_embedding = UserTagEmbedding(path, dataset=self.dataset_name, embed_size=self.embedding_size,
                                              feature_dim=self.user_attribute_dim)
        self.embedding_usertag_index, self.user_attribute_num = User_tag_embedding.user_tag_index4embed(min_count=10)
        print('Total user personal information features:', self.user_attribute_num)

    def _load_item2vec_feature(self, path):
        print('- loading all feature embedding matrix...')
        Feature_embedding = FeatureEmbedding(path, embed_size=self.embedding_size, name=[], dataset=self.dataset_name)  # V5
        self.embedding_feature_index, self.feature_dims, _ = Feature_embedding.feature_index4embed(train_w2v=False)
        self.feature_num = [Feature_embedding.feature_num[i] for i in Feature_embedding.feature_name]
        print('Total feature dims:', self.feature_dims)

    def build_graph(self):
        # create item embedding matrix
        if self.embed_type == 'feature':
            self._load_item2vec_feature(args.path)
        if self.user_attribute_ratio:
            # create user tag embedding matrix
            self._load_user_info(args.path)

        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")


def training(model, dataset, epochs, num_negatives, topK, memory_length):
    global istraining
    if not os.path.exists('Prediction'):
        os.mkdir('Prediction')

    config = tf.ConfigProto(device_count={"GPU": 1})
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        logging.info("Variables initialized")
        print("Variables initialized")

        bg = batchGenerate.batchGen()
        # initialization for training batches
        batch_begin = time()
        batches = bg.shuffle(dataset, model.batch_size, num_negatives, memory_length)
        batch_time = time() - batch_begin
        num_batch = len(batches[1])
        batch_index = list(range(num_batch))

        testDict = evaluate.init_evaluate_model(model, sess, dataset.testList, dataset.trainList, memory_length)

        # train by epoch
        for epoch_count in range(epochs):
            train_begin = time()
            training_batch(batch_index, model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % model.verbose == 0:
                if model.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0

                # evaluation
                eval_begin = time()
                hits, ndcgs, aps, rs, losses = evaluate.eval(model, sess, dataset.testList, testDict, topK)
                for i in range(len(topK)):
                    hr, ndcg, ap, r, test_loss = hits[i].mean(), np.array(ndcgs[i]).mean(), \
                                                 np.array(aps[i]).mean(), np.array(rs[i]).mean(), np.array(losses[i]).mean()
                    eval_time = time() - eval_begin
                    istraining = False

                    logging_result = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, AP = %.4f, R = %.4f, " \
                                     "loss = %.4f [%.1fs] train_loss = %.4f [%.1fs], topk = %d" % (
                        epoch_count, batch_time, train_time, hr, ndcg, ap, r,
                        test_loss, eval_time, train_loss, loss_time, topK[i])

                    logging.info(logging_result)
                    if i == len(topK) - 1:
                        print(logging_result)

                istraining = True

            batch_begin = time()
            batches = bg.shuffle(dataset, model.batch_size, num_negatives, memory_length)
            np.random.shuffle(batch_index)
            batch_time = time() - batch_begin


def training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_id, user_input, num_idx, item_input, labels = batchGenerate.batch_gen(batches, index)
        feed_dict = {model.user_id: user_id[:, None],
                     model.user_input: user_input,
                     model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None],
                     model.labels: labels[:, None]}
        sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])

    for index in range(num_batch):
        user_id, user_input, num_idx, item_input, labels = batchGenerate.batch_gen(batches, index)
        feed_dict = {model.user_id: user_id[:, None],
                     model.user_input: user_input,
                     model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None],
                     model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


if __name__ == '__main__':
    args = parse_args()
    regs = eval(args.regs)

    log_dir = "Log/%s/" % args.dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = strftime('%Y_%m_%d_%H-%M', localtime())
    log_filename = os.path.join(log_dir, "%s_s%.1f_lr%.0e_p%d_%s_m%d_u%.1f_%s" %
                                (args.embed_type, args.sparsity, args.lr, args.prediction,
                                 args.feature_merge, args.memory, args.user_gamma, timestamp))
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info("begin training KAC model ......")
    logging.info(args)
    print(args)

    test_pos_num, test_neg_num = args.pos_neg.split(':')
    dataset = Dataset(args.path + args.dataset, int(test_pos_num), int(test_neg_num), sparsity=args.sparsity)

    model = KAC(dataset.num_items, args)
    model.build_graph()
    training(model, dataset, args.epochs, args.num_neg,
             list(map(int, args.topk.split(','))),  args.memory)
