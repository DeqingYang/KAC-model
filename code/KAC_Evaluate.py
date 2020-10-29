
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import pickle

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_topK = None
_DictList = None
_sess = None
file_log_prediction = None


def init_evaluate_model(model, sess, testList, trainList, memory_length=0):
    """
    Evaluate the performance (HR, NDCG, MAP, MRR) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testList
    global _trainList
    global _DictList
    global _sess
    global _memory
    global _epoch

    _sess = sess
    _model = model
    _testList = testList
    _trainList = trainList
    _memory = memory_length
    _epoch = 0

    return load_test_as_list()


def eval(model, sess, testList, DictList, topK):
    global _model
    global _testList
    global _testNegatives
    global _topK
    global _DictList
    global _sess
    global _epoch
    _model = model
    _testList = testList
    _DictList = DictList
    _sess = sess
    _topK = topK

    HR, NDCG, AP, RR, LOSS = [], [], [], [], []  # [hr@1, hr@3, hr@5]
    hits, ndcgs, aps, rrs, losses = [], [], [], [], []  # [[e@1, e@3, e@5], []...]

    if file_log_prediction:
        file_log_prediction.writelines('Epoch %d\n' % _epoch)
    _epoch += 1

    for user_id in _testList.keys():
        hr, ndcg, ap, rr, loss = _eval_one_rating(user_id)
        hits.append(hr)
        ndcgs.append(ndcg)
        aps.append(ap)
        rrs.append(rr)
        losses.append(loss)

    for i in range(len(_topK)):
        HR.append(np.array(hits)[:, i])
        NDCG.append(np.array(ndcgs)[:, i])
        AP.append(np.array(aps)[:, i])
        RR.append(np.array(rrs)[:, i])
        LOSS.append(np.array(losses)[:, i])

    return HR, NDCG, AP, RR, LOSS


def load_test_as_list():
    DictList = {}
    for user, value in _testList.items():
        pos, neg = value
        user_train_records = _trainList[user]
        num_idx_ = len(user_train_records)
        if _memory and _memory < num_idx_:  # memory-enhanced attention mechanism
            num_idx_ = _memory
            user_train_records = list(reversed(user_train_records[-_memory:]))
        else:
            # notice that we do not apply zero padding to memory
            user_train_records = list(reversed(user_train_records))

        test_items = neg + pos
        user_id = np.array([user] * len(test_items))[:, None]
        # Get prediction scores
        num_idx = np.full(len(test_items), num_idx_, dtype=np.int32)[:, None]
        user_input = np.array([user_train_records] * len(test_items))
        item_input = np.array(test_items)[:, None]
        feed_dict = {_model.user_id: user_id,
                     _model.user_input: user_input,
                     _model.num_idx: num_idx,
                     _model.item_input: item_input}
        DictList[user] = feed_dict
    print("already load the evaluate model...")
    return DictList


def _eval_one_rating(idx):
    global _epoch
    pos, neg = _testList[idx]
    labels = [0] * len(neg) + [1] * len(pos)
    labels = np.array(labels)[:, None]
    feed_dict = _DictList[idx]
    feed_dict[_model.labels] = labels

    try:
        _model.A
    except:
        user_p, predictions, loss = _sess.run([_model.embedding_p, _model.output, _model.loss], feed_dict=feed_dict)
    else:
        user_p, predictions, loss, attention_mat = _sess.run([_model.embedding_p, _model.output, _model.loss, _model.A],
                                                             feed_dict=feed_dict)

    # sort
    _pred_index = list(zip(predictions, np.arange(len(labels))))
    _pred_sort = sorted(_pred_index, key=lambda x: x[0], reverse=True)
    pos_position = np.array([_pred_sort.index(i) for i in _pred_sort if i[1] >= len(neg)])

    HR, NDCG, AP, RR, LOSS = [], [], [], [], []
    for k in _topK:
        # calculate HR@K, NDCG@K, MAP@K, MRR@K
        hit, dcg, idcg = 0, 0, 0
        ap, rr = 0, 0

        for loc in pos_position:
            if loc < k:
                hit += 1
                dcg += 1 / math.log(loc + 2)
                idcg += 1 / math.log(hit + 1)
                ap += hit / (loc + 1)

        hr = hit / k
        ndcg = dcg / idcg if hit != 0 else 0
        ap = ap / k
        rr = 1 / (min(pos_position) + 1)

        HR.append(hr)
        NDCG.append(ndcg)
        AP.append(ap)
        RR.append(rr)
        LOSS.append(loss)

    return HR, NDCG, AP, RR, LOSS
