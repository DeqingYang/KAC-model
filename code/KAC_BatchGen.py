import multiprocessing
import numpy as np


def batch_gen(batches, i):
    return [(batches[r])[i] for r in range(5)]


def _remove_item(users, item):
    try:
        i = users.index(item)
        users[i] = users[-1]
        users[-1] = 0  # feature_mask
        flag = 1
    except:
        flag = 0
    return len(users) - flag


def _add_mask(features, num_max):
    # uniformalize the length of each batch
    for i in range(len(features)):
        features[i] = features[i] + [0] * (num_max + 1 - len(features[i]))
    return features


def _get_sequence(users, item, memory_length):
    try:  # positive training example
        i = users.index(item)
    except:  # negative training example
        return [], -1
    else:
        if memory_length and i >= memory_length:  # memory mechanism
            memory = users[i-memory_length:i]
        else:
            # use all history without memory mechanism or not enough history records
            # notice that we do not apply zero padding to memory
            memory = users[:i]
    return memory, i


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
        self.memory_length = 0
        self.min_history = 2

    def shuffle(self, dataset, batch_size, num_negatives, memory_length):
        # negative sampling and shuffle the data
        self._Dataset = dataset
        self._num_negatives = num_negatives
        self.memory_length = memory_length
        self._batch_size = int(batch_size)
        self._num_items, self._user_input, self._item_input, self._labels = self._get_train_data_fixed()
        # print('Negative samples done!')
        iterations = len(self._user_input)
        self._index = np.arange(iterations)
        self._num_batch = iterations // self._batch_size
        data_input = self._preprocess(self._get_train_batch_fixed)

        return data_input

    def _preprocess(self, get_train_batch):  # generate the masked batch list
        user_id_list = []
        user_input_list, num_idx_list, item_input_list, labels_list = [], [], [], []
        cpu_count = 1  # multiprocessing.cpu_count()
        if cpu_count == 1:
            for i in range(self._num_batch):
                uid, ui, ni, ii, l = get_train_batch(i)
                user_id_list.append(uid)
                user_input_list.append(ui)
                num_idx_list.append(ni)
                item_input_list.append(ii)
                labels_list.append(l)
        else:
            pool = multiprocessing.Pool(cpu_count)
            res = pool.map(get_train_batch, range(self._num_batch))
            pool.close()
            pool.join()
            user_id_list = [r[0] for r in res]
            user_input_list = [r[1] for r in res]
            num_idx_list = [r[2] for r in res]
            item_input_list = [r[3] for r in res]
            labels_list = [r[4] for r in res]
        return user_id_list, user_input_list, num_idx_list, item_input_list, labels_list

    def _get_train_data_fixed(self):
        user_input, item_input, labels = [], [], []
        trainList = self._Dataset.trainList
        testList = self._Dataset.testList
        num_items = self._Dataset.num_items - 1
        for u, items in trainList.items():
            for i in items:
                # positive instance
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                # negative instances
                for t in range(self._num_negatives):
                    j = np.random.randint(num_items)

                    while j in trainList[u] + testList[u][0] + testList[u][1]:
                        j = np.random.randint(num_items)
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)

        return num_items, user_input, item_input, labels

    def _get_train_batch_fixed(self, i):
        # represent the feature of users via items rated by him/her
        user_id = []
        user_list, num_list, item_list, labels_list = [], [], [], []
        trainList = self._Dataset.trainList
        begin = i * self._batch_size
        for idx in range(begin, begin + self._batch_size):
            user_idx = self._user_input[self._index[idx]]
            item_idx = self._item_input[self._index[idx]]
            nonzero_row = []
            nonzero_row += trainList[user_idx]

            memory, loc = _get_sequence(nonzero_row, item_idx, self.memory_length)
            if loc < 0 and num_list != []:  # negative
                user_id.append(user_idx)
                num_list.append(num_list[-1])
                user_list.append(user_list[-1])
                item_list.append(item_idx)
                labels_list.append(self._labels[self._index[idx]])

            if loc >= self.min_history:

                if self.memory_length != 0 and loc > self.memory_length:
                    num = self.memory_length
                else:
                    num = loc
                user_id.append(user_idx)
                num_list.append(num)
                user_list.append(list(reversed(memory)))
                item_list.append(item_idx)
                labels_list.append(self._labels[self._index[idx]])

        user_id = np.array(user_id)
        user_input = np.array(_add_mask(user_list, max(num_list)))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(labels_list)

        return user_id, user_input, num_idx, item_input, labels
