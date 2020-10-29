import multiprocessing
import numpy as np


# _Dataset = None
# _batch_size = None
# _num_negatives = None
# self._num_items = None
# self._user_input = None
# self._item_input = None
# self._labels = None
# self._index = None
# self._num_batch = None
# self._batch_length = None


def batch_gen(batches, i):
    return [(batches[r])[i] for r in range(4)]


def _remove_item(feature_mask, users, item):
    flag = 0
    for i in range(len(users)):
        if users[i] == item:
            users[i] = users[-1]
            users[-1] = feature_mask
            flag = 1
            break
    return len(users) - flag


def _add_mask(feature_mask, features, num_max):
    # uniformalize the length of each batch
    for i in range(len(features)):
        features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
    return features


class batchGen:
    def __init__(self):
        self._Dataset = None
        self._batch_size = None
        self._num_negatives = None
        self._num_items = None
        self._user_input = None
        self._item_input = None
        self._labels = None
        self._index = None
        self._num_batch = None
        self._batch_length = None

    def shuffle(self, dataset, batch_choice, num_negatives):  # negative sampling and shuffle the data

        # global _Dataset
        # global _batch_size
        # global _num_negatives
        # global self._num_items
        # global self._user_input
        # global self._item_input
        # global self._labels
        # global self._index
        # global self._num_batch
        # global self._batch_length
        self._Dataset = dataset
        self._num_negatives = num_negatives

        if batch_choice == 'user':
            self._num_items, self._user_input, self._item_input, self._labels, self._batch_length = self._get_train_data_user()
            self._num_batch = len(self._batch_length)
            return self._preprocess(self._get_train_batch_user)

        else:
            batch_choices = batch_choice.split(":")
            if batch_choices[0] == 'fixed':
                self._batch_size = int(batch_choices[1])
                self._num_items, self._user_input, self._item_input, self._labels = self._get_train_data_fixed()
                print('Negative samples done!')
                iterations = len(self._user_input)
                self._index = np.arange(iterations)
                self._num_batch = iterations // self._batch_size  # ???

                data_input = self._preprocess(self._get_train_batch_fixed)
                # print('data_input(user_input_list, num_idx_list, item_input_list, labels_list):',
                #       data_input[0][:2], data_input[1][:2], data_input[2][:2], data_input[3][:2])
                return data_input
            else:
                print("invalid batch size !")

    def _preprocess(self, get_train_batch):  # generate the masked batch list
        user_input_list, num_idx_list, item_input_list, labels_list = [], [], [], []
        cpu_count = 1  # multiprocessing.cpu_count()
        if cpu_count == 1:
            for i in range(self._num_batch):
                ui, ni, ii, l = get_train_batch(i)
                user_input_list.append(ui)
                num_idx_list.append(ni)
                item_input_list.append(ii)
                labels_list.append(l)
        else:
            pool = multiprocessing.Pool(cpu_count)
            res = pool.map(get_train_batch, [i for i in range(self._num_batch)])  # range(self._num_batch)
            pool.close()
            pool.join()
            res = list(map(get_train_batch, range(self._num_batch)))   # adjusted
            user_input_list = [r[0] for r in res]
            num_idx_list = [r[1] for r in res]
            item_input_list = [r[2] for r in res]
            labels_list = [r[3] for r in res]
        return user_input_list, num_idx_list, item_input_list, labels_list

    def _get_train_data_user(self):
        user_input, item_input, labels, batch_length = [], [], [], []
        train = self._Dataset.trainMatrix
        trainList = self._Dataset.trainList
        num_items = train.shape[1] - 1
        num_users = train.shape[0] - 1
        for u in range(num_users):
            if u == 0:
                batch_length.append((1 + self._num_negatives) * len(trainList[u]))
            else:
                batch_length.append((1 + self._num_negatives) * len(trainList[u]) + batch_length[u - 1])
            for i in trainList[u]:
                # positive instance
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                # negative instances
                for t in range(self._num_negatives):
                    j = np.random.randint(num_items)
                    while j in trainList[u]:
                        j = np.random.randint(num_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
        return num_items, user_input, item_input, labels, batch_length

    def _get_train_batch_user(self, i):
        # represent the feature of users via items rated by him/her
        user_list, num_list, item_list, labels_list = [], [], [], []
        trainList = self._Dataset.trainList
        if i == 0:
            begin = 0
        else:
            begin = self._batch_length[i - 1]
        batch_index = list(range(begin, self._batch_length[i]))
        np.random.shuffle(batch_index)
        for idx in batch_index:
            user_idx = self._user_input[idx]
            item_idx = self._item_input[idx]
            nonzero_row = []
            nonzero_row += trainList[user_idx]
            num_list.append(_remove_item(self._num_items, nonzero_row, item_idx))
            user_list.append(nonzero_row)
            item_list.append(item_idx)
            labels_list.append(self._labels[idx])

        user_input = np.array(_add_mask(self._num_items, user_list, max(num_list)))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(labels_list)
        return (user_input, num_idx, item_input, labels)

    def _get_train_data_fixed(self):
        print('Negative sampling...')
        user_input, item_input, labels = [], [], []
        trainList = self._Dataset.trainList

        train = self._Dataset.trainMatrix
        num_items = train.shape[1] - 1
        for (u, i) in train.keys():
            # positive instance
            user_items = []
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(self._num_negatives):
                j = np.random.randint(num_items)
                while j in trainList[u]:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)

        return num_items, user_input, item_input, labels

    def _get_train_batch_fixed(self, i):
        # represent the feature of users via items rated by him/her
        user_list, num_list, item_list, labels_list = [], [], [], []
        trainList = self._Dataset.trainList
        begin = i * self._batch_size
        for idx in range(begin, begin + self._batch_size):
            user_idx = self._user_input[self._index[idx]]
            item_idx = self._item_input[self._index[idx]]
            nonzero_row = []
            nonzero_row += trainList[user_idx]
            num_list.append(_remove_item(self._num_items, nonzero_row, item_idx))
            user_list.append(nonzero_row)
            item_list.append(item_idx)
            labels_list.append(self._labels[self._index[idx]])

        user_input = np.array(_add_mask(self._num_items, user_list, max(num_list)))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(labels_list)
        return user_input, num_idx, item_input, labels
