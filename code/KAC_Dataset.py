# coding=utf-8

import scipy.sparse as sp
import numpy as np
from time import time

import pandas as pd
from gensim import models, corpora, similarities
import random


def dict_map(dic, x):
    if x in dic.keys():
        return dic[x]
    else:
        return 0


class FeatureEmbedding:
    def __init__(self, path, embed_size, name, dataset):
        self.dataset = dataset
        if dataset == 'douban':
            self.df_item = pd.read_csv(path + 'movie_data.csv')
            self.feature_name = ['type', 'tags', 'directors', 'writers', 'actors', 'countries', 'year'] if name == [] else name
            self.feature_num = {'type': 3, 'tags': 8, 'directors': 1, 'writers': 2,
                                'actors': 8, 'countries': 1, 'year': 1}
        if dataset == 'music':
            self.df_item = pd.read_csv(path + 'music_data.csv')
            self.feature_name = ['singer', 'album', 'composer', 'author', 'playlist'] if name == [] else name  #
            self.feature_num = {'singer': 1, 'album': 1, 'composer': 1, 'author': 1, 'playlist': 3}  #

        self.index = 0
        self.embed_size = embed_size

    def feature2dict(self, input_list, split_symbol):
        dictionary = {}
        dictionary['None'] = 0
        for x in input_list:
            if pd.isnull(x):
                dictionary['None'] += 1
                continue

            if split_symbol in x:
                multi = x.split(split_symbol)
                for single in multi:
                    if single in dictionary.keys():
                        dictionary[single] += 1
                    else:
                        dictionary[single] = 1
            else:
                if x in dictionary.keys():
                    dictionary[x] += 1
                else:
                    dictionary[x] = 1
        return dictionary

    def word2vec_training(self, lst, split_symbol):
        texts = []
        for line in lst:
            if pd.isnull(line):
                items = ['None']
            else:
                items = line.split(split_symbol)
            texts.append(items)
        # dictionary = corpora.Dictionary(texts, prune_at=2000000)
        model = models.word2vec.Word2Vec(texts, size=self.embed_size, window=5, min_count=1, negative=3)
        return texts, model

    def build_embedding_matrix(self, lst, w2v_model, coeff=0.005):
        first_row = np.zeros(self.embed_size).astype(np.float32)
        embedding_matrix = []
        for item in lst:
            value = sum(list(map(lambda x: w2v_model[x], item))) / len(item)
            embedding_matrix.append(value)
        scale = coeff / np.var(embedding_matrix)
        embedding_matrix_scaled = list(map(lambda x: scale * x, embedding_matrix))
        matrix_scaled = np.vstack((first_row, embedding_matrix_scaled))
        return matrix_scaled

    def generate_feature2index_dict(self, count_dict, min_count=5):
        self.index += 1   # relate to feature_id = 0, for zero padding
        index_dict = {}
        sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        for (feature, count) in sorted_count_dict:
            if count >= min_count:
                index_dict[feature] = self.index
                self.index += 1
        return index_dict, self.index

    def feature2index(self, lst, index_dict, split_symbol, max_feature_num=5):
        index_list = [[0] * max_feature_num]  # relate to item_id = 0, which means no item
        for line in lst:
            if pd.isnull(line):
                line = 'None'
            line = line.split(split_symbol)
            features = list(map(lambda x: dict_map(index_dict, x), line))

            if len(features) < max_feature_num:
                features += [0] * (max_feature_num - len(features))
            else:
                features = features[:max_feature_num]
            index_list.append(features)

        return index_list

    def feature_encode(self, lst, split_symbol, max_feature_num=5, w2v=False):
        _feature_dict = self.feature2dict(lst, split_symbol)
        _feature_index_dict, _max_index = self.generate_feature2index_dict(_feature_dict)
        _feature_index4embed = self.feature2index(lst, _feature_index_dict, split_symbol, max_feature_num)
        if w2v:
            _texts, _model = self.word2vec_training(lst, split_symbol)
            _mat = self.build_embedding_matrix(_texts, _model)
        else:  # random initialize those feature embedding matrix only one in each item
            _mat = np.random.normal(loc=0.0, scale=0.01, size=[len(_feature_index4embed), self.embed_size])

        return _feature_index4embed, _max_index, _mat.tolist()  # V5

    def feature_index4embed(self, train_w2v=False):
        index4embed = []
        max_index = []
        w2v_embed_matrix = []
        # type
        if self.dataset == 'douban':
            _symbol_dict = {'type': ' / ', 'tags': '#', 'directors': ' / ', 'writers': ' / ',
                            'actors': ' / ', 'countries': ' / ', 'year': ' '}
            _w2v_dict = {'type': 1, 'tags': 1, 'directors': 0, 'writers': 0,
                         'actors': 1, 'countries': 0, 'year': 0}

        if self.dataset == 'music':
            _symbol_dict = {'singer': ' ', 'album': ' ', 'composer': '/', 'author': '/', 'playlist': '-'}
            _w2v_dict = {'singer': 0, 'album': 0, 'composer': 0, 'author': 0, 'playlist': 0}

        for dim in self.feature_name:
            _index4embed, _max_index, _embed_matrix = self.feature_encode(lst=self.df_item[dim],
                                                                          split_symbol=_symbol_dict[dim],
                                                                          max_feature_num=self.feature_num[dim],
                                                                          w2v=train_w2v and _w2v_dict[dim])
            index4embed.append(_index4embed)
            max_index.append(_max_index)
            w2v_embed_matrix += _embed_matrix

        index4embed = np.concatenate(index4embed, axis=-1)
        return index4embed, max_index, w2v_embed_matrix


class UserTagEmbedding:
    def __init__(self, path, dataset, embed_size, feature_dim):
        global _attribute
        self.dataset = dataset
        self.embed_size = embed_size  # difference between V4, V5
        self.feature_dim = feature_dim

        if self.dataset == 'douban':
            self.df_user = pd.read_csv(path + 'user_tags.csv')
            _attribute = 'tags'
        if self.dataset == 'music':
            self.df_user = pd.read_csv(path + 'user_data.csv')
            _attribute = 'city,province,birthday'
        if self.dataset == 'food':
            pass

    def user_tag2dict(self, input_list, min_count=5):
        count_dict = {}
        for x in input_list:
            for x_i in x:
                if x_i in count_dict.keys():
                    count_dict[x_i] += 1
                else:
                    count_dict[x_i] = 1
        index = 1  # relate to user_feature_id = 0, for zero padding
        index_dict = {}
        sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        for (feature, count) in sorted_count_dict:
            if count >= min_count:
                index_dict[feature] = index
                index += 1
        return index_dict, index  # , count_dict

    def user_tag2index(self, lst, index_dict):
        index_list = []
        for line in lst:
            features = list(map(lambda x: dict_map(index_dict, x), line))
            if len(features) < self.feature_dim:
                features += [0] * (self.feature_dim - len(features))
            else:
                features = features[:self.feature_dim]
            index_list.append(features)
        return index_list

    def user_tag_index4embed(self, min_count=5):
        if self.dataset == 'douban':
            user_info = [str(line).split(' ') if not pd.isnull(line) else []
                         for line in self.df_user[_attribute]]
        if self.dataset == 'music':
            user_info = []
            for i in range(len(self.df_user)):
                line = self.df_user.iloc[i]
                user_info.append(list(self.df_user.iloc[i][_attribute.split(',')].values) if not pd.isnull(line.all()) else [])

        user_index_dict, max_index = self.user_tag2dict(user_info, min_count=min_count)
        user_index4embed = self.user_tag2index(user_info, user_index_dict)
        return np.array(user_index4embed), max_index


class Dataset(object):
    def __init__(self, path, test_pos_num=5, test_neg_num=95, sparsity=False):
        self.num_users, self.num_items = 0, 0
        self.test_pos_num, self.test_neg_num = test_pos_num, test_neg_num
        self.sparsity = sparsity
        self.popluar_items = []
        self.cold_start_test_items = []
        self.cold_start_user = []
        # trainList=[[item id ...][...]...], testList=[[user, [test pos...], [test neg...]], ...]
        self.ratingList_raw = self.load_training_file_as_list(path + ".train.rating")
        self.ratingList = self.history_orderByTime()  # order by time, sequential
        self.trainList, self.testList = self.split_train_test_items()

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        u_ = 0
        lists, items = {}, []
        item_count_dict = {}
        with open(filename, "r") as f:
            lines = f.readlines()
            index = 0

        for line in lines:
            if line is not None and line != "":
                arr = line.strip('\n').split("\t")
                u, i,_, t = list(map(int, arr))
				#u,i,rate,_,t = list(map(int,arr))
                num_users = max(num_users, u)
                num_items = max(num_items, i)

                if u_ < u:
                    index = 0
                    if len(items) >= self.test_pos_num:
                        lists[u_] = items
                    items = []
                    u_ += 1
                index += 1
				#if rate > :
                items.append((i, t))
                # count the item frequency
                if i in item_count_dict.keys():
                    item_count_dict[i] += 1
                else:
                    item_count_dict[i] = 1

        if len(items) >= self.test_pos_num:  # only keep the users with at least 'test_pos_num' records
            lists[u_] = items
        self.num_users, self.num_items = num_users + 1, num_items + 1
        # get the popular items
        item_count_dict_sorted = sorted(item_count_dict.items(), key=lambda x: x[1], reverse=True)
        topPopluar = int(len(item_count_dict) * 0.1)
        self.popluar_items = [i[0] for i in item_count_dict_sorted[:topPopluar]]
        print('Already load the trainList, total users: %d, total items: %d' % (self.num_users, self.num_items))
        print('Popular items: %d' % len(self.popluar_items))
        return lists

    def split_train_test_items(self):
        item_in_train, item_in_test = set(), set()
        train_lists, train_items = {}, []
        test_lists, test_items = {}, []
        neg_items = []
		neg_items_train = []
        total_training_examples = 0
        for u, items_list in self.ratingList.items():
            # negtive test sampling
            while len(neg_items) < self.test_neg_num:
                # sample from all items if the records of user exceeds popular item number
                if len(items_list) > len(self.popluar_items):
                    neg_itemid = random.sample(list(range(1, self.num_items)), 1)[0]
                    # neg_itemid = np.random.randint(self.num_items)
                else:
                    neg_itemid = random.sample(self.popluar_items, 1)[0]
                if neg_itemid not in items_list and neg_itemid not in neg_items:
                    neg_items.append(neg_itemid)

            # Sequential
            X_train, X_test = items_list[:-self.test_pos_num], items_list[-self.test_pos_num:]

            if self.sparsity != 0:
                train_pos_num = round(self.sparsity * len(X_train))
				train_neg_num = train_pos_num * 5
                if train_pos_num <= 1:  # ensure that every user has at least 2 records
                    self.cold_start_user.append(u)
                    train_items = []
                else:
                    train_items = random.sample(X_train, train_pos_num)
                    train_items = sorted(train_items, key=items_list.index)  # reorder the item sequence
				
				while len(neg_items_train) < train_neg_num:
					if len(items_list) > len(self.popular_items):
						neg_itemid = random.sample(list(range(1, self.num_items)), 1)[0]
					else:
						neg_itemid = random.sample(self.popluar_items, 1)[0]
					if neg_itemid not in items_list and neg_itemid not in neg_items_train:
						neg_items_train.append(neg_itemid)
				
                item_in_train.update(train_items)
                item_in_test.update(X_test)
                train_lists[u] = [train_items,neg_items_train]
                test_lists[u] = [X_test, neg_items]
                total_training_examples += len(train_items)
                train_items, test_items, neg_items = [], [], []  # dump

            else:
                train_lists[u] = [train_items,neg_items_train]
                test_lists[u] = [X_test, neg_items]
                total_training_examples += len(train_items)
                train_items, test_items, neg_items = [], [], []  # dump

        self.cold_start_test_items = list(item_in_test - item_in_train)
        print('total training examples: %d' % total_training_examples)
        print('cold start users: %d, cold start items: %d' % (
        len(self.cold_start_user), len(self.cold_start_test_items)))
        return train_lists, test_lists

    def history_orderByTime(self):
        '''
        self.ratingList_raw = [[(item, timestamp), ()], [()]...]
            each line in the raw rating list is a user's history: (item, timestamp)
        '''
        ratingList = {}
        for u, user_item_interaction in self.ratingList_raw.items():
            orderedRatings = sorted(user_item_interaction, key=lambda i: i[1])  # order by time
            sequentialItems = [r[0] for r in orderedRatings]
            ratingList[u] = sequentialItems
        return ratingList