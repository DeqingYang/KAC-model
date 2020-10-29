#!/usr/bin/python3

"""
load data set
"""
import sys
import os
import re
import random
import pandas as pd
import time
# import numpy as np
# import copy


class Movielens_1m:
    def __init__(self, root='Data/ml-1m/', train=None, test=None):
        self.root = root
        self.ratingList = self.load_rating_data(root + '/ratings.dat')  # [uid, iid, rating, timestamp]
        self.userList = self.load_user_data(root + '/users.dat')        # [uid, gender, age, occupation, zipcode]
        self.itemList = self.load_item_data(root + '/movies.dat')       # [iid, title, genres]
        self.userNum = len(self.userList)
        self.itemNum = len(self.itemList)
        self.train = train
        self.test = test
        self.trainList = []  # [uid, iid, rating, timestamp]
        self.testList = []
        self.trainDict = {}
        self.testDict = {}
        # from original NAIS
        self.testNegatives = self.load_negative_file(root + "/ml-1m.test.negative")

        # for memory mechanism
        self.userDict = {}          # {user: [[iid], [rating], [timestamp]]}
        self.train_memory_pos = []  # [(user, memories, iid)]
        self.test_memory_pos = []   # [(user, memories, iid)]

        # for negative sample, [(user, memories, iid, label)]
        self.train_memory_neg = []
        self.train_memory_merge = []
        self.test_memory_neg = []
        self.test_memory_merge = []

    # from original NAIS
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = list(map(int, arr[1:]))
                # saved in form: str(user, item_pos)
                userid = arr[0][1:].split(',')[0]
                negativeList.append((int(userid), [], negatives, 0))
                line = f.readline()
        return negativeList

    def load_rating_data(self, path):
        """
        load movie lens 1M ratings from original rating file.
        need to download and put rating data in /data folder first.
        Source: http://www.grouplens.org/
        """
        # users = pd.read_csv(upath, sep='::', header=None, names=unames, encoding=encoding)
        # ratings = pd.read_csv(rpath, sep='::', header=None, names=rnames, encoding=encoding)
        # movies = pd.read_csv(mpath, sep='::', header=None, names=mnames, encoding=encoding)
        _ratingList = []
        t1 = time.time()
        fp = open(path, 'r')
        for line in fp:
            userid, itemid, rating, timestamp = line.strip('\n').split('::')
            _ratingList.append([int(userid), int(itemid), float(rating), int(timestamp)])
        fp.close()
        t2 = time.time()
        print('Load rating data success! Total ratings: %d, Time: %.2fms' % (len(_ratingList), (t2-t1) * 1000),
              file=sys.stderr)
        return _ratingList

    def load_user_data(self, path):
        _userList = []
        t1 = time.time()
        fp = open(path, 'r')
        for line in fp:
            uid, gender, age, occupation, zipcode = line.strip('\n').split('::')
            _userList.append([int(uid), gender, int(age), int(occupation), zipcode])
        fp.close()
        t2 = time.time()
        print('Load user data success! Total users: %d, Time: %.2fms' % (len(_userList), (t2-t1) * 1000),
              file=sys.stderr)
        return _userList

    def load_item_data(self, path):
        _itemList = []
        t1 = time.time()
        fp = open(path, 'r', encoding='latin1')
        for line in fp:
            iid, title, genres = line.strip('\n').split('::')
            _itemList.append([int(iid), title, genres])
        fp.close()
        t2 = time.time()
        print('Load item data success! Total items: %d, Time: %.2fms' % (len(_itemList), (t2-t1) * 1000),
              file=sys.stderr)
        return _itemList

    def spilt_training_test(self, trainSize=0.8):
        '''split ratings to training set and test set '''
        for line in self.ratingList:
            rand = random.random()
            if rand < trainSize:
                self.trainList.append(line)
            else:
                self.testList.append(line)
        # shuffle?
        print('Split training set and test set success!', file=sys.stderr)

    def getDict(self):
        for line in self.ratingList:
            userid, itemid, record, _ = line
            self.trainDict.setdefault(userid, {})
            self.trainDict[userid][itemid] = record
        for line in self.trainList:
            userid, itemid, record, _ = line
            self.trainDict.setdefault(userid, {})
            self.trainDict[userid][itemid] = record
        for line in self.testList:
            userid, itemid, record, _ = line
            self.testDict.setdefault(userid, {})
            self.testDict[userid][itemid] = record

    def getDataframe(self):
        '''run self.init() before using this function'''
        rating_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
        user_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
        item_title = ['MovieID', 'Title', 'Genres']
        ratings = pd.DataFrame(data=self.ratingList, columns=rating_title)
        users = pd.DataFrame(data=self.userList, columns=user_title)
        items = pd.DataFrame(data=self.itemList, columns=item_title)
        return ratings, users, items

    def memory_orderByTime(self):
        '''
        Initialize after 'self.load_rating_data()'
        self.ratingList = [userid, movieid, rating, timestamp]
        self.userDict = {user: [uMovieid], [uRating], [uTimestamp]}
        '''
        uTmp = []
        last_userid = self.ratingList[0][0]  # id of first user
        for userid, itemid, rating, timestamp in self.ratingList:
            if userid != last_userid:
                # all recordings of one user are collected in uTmp
                orderedRatings = sorted(uTmp, key=lambda i: i[2])  # order by time
                uItemid = [r[0] for r in orderedRatings]
                uRating = [r[1] for r in orderedRatings]
                uTimestamp = [r[2] for r in orderedRatings]
                self.userDict[last_userid] = [uItemid, uRating, uTimestamp]
                last_userid = userid  # switch to next user
                uTmp = []  # Clearup
                uTmp.append((itemid, rating, timestamp))
            else:
                uTmp.append((itemid, rating, timestamp))
        # last user
        orderedRatings = sorted(uTmp, key=lambda i: i[2])  # order by time
        uItemid = [r[0] for r in orderedRatings]
        uRating = [r[1] for r in orderedRatings]
        uTimestamp = [r[2] for r in orderedRatings]
        self.userDict[userid] = [uItemid, uRating, uTimestamp]

        return self.userDict

    def memory_generate_train_withOneItemOut(self, MEMORY_CAP=30, MEMORY_MIN=1, RATING_MIN=20):
        t1 = time.time()
        '''
        :param MEMORY_CAP: the capacity of memory, default=30
        :param MEMORY_MIN: minimum ratings of memory for a training sample, default=1
        :param RATING_MIN: minimun ratings of a single user, filter those with few history records, default=20
        :return: training and test data for memory mechanism
        
        @ here exits the problem that more than one ratings at a time @
        '''
        # last_user = list(dictionary.keys())[0]  # first user
        for user, ratings in self.userDict.items():
            itemid, rating, timestamp = ratings
            # Filter users with less than RATING_MIN
            if len(itemid) <= RATING_MIN:
                continue
            else:
                # uniform length of memory items
                memories = [self.itemNum] * MEMORY_CAP
                # use a queue structure, first in first out
                memo = itemid[:MEMORY_MIN]
                memo.reverse()
                memories[:MEMORY_MIN] = memo
                for index, iid in enumerate(itemid[MEMORY_MIN:-1]):
                    self.train_memory_pos.append((user, memories.copy(), iid, rating[index + MEMORY_MIN]))
                    memories.pop()  # remove last
                    memories.insert(0, iid)  # insert in the head
                # Leave the last item as test data
                self.test_memory_pos.append((user, memories.copy(), itemid[-1], rating[index + MEMORY_MIN]))
        t2 = time.time()

        print('Generate memory training and test set success! Time: %.2fms' % ((t2-t1) * 1000), file=sys.stderr)
        return self.train_memory_pos, self.test_memory_pos

    def memory_negative_sampling(self, NEGATIVE_NUM=4):
        t1 = time.time()
        test_neg_dic = {}
        last_user = -1  # self.test_memory_pos[0][0]  # first user id
        itemList = [i[0] for i in self.itemList]
        print('Start negative sampling...', file=sys.stderr)

        for user, memories, itemid, rating in self.train_memory_pos:
            if user != last_user:
                # Refresh itemList for next user
                user_itemlist = itemList.copy()
                random.shuffle(user_itemlist)
                last_user = user
                # Add the rest negative items to test set

                # test_tmp = []
                # while not user_itemlist == []:  # before pop out, we should check whether the user_itemlist is empty
                #     negativeItemid = user_itemlist.pop()
                #     if not negativeItemid in self.userDict[user][0]:
                #         test_tmp.append(negativeItemid)

                # use set.difference() to remove positive items from rest items
                test_neg_dic[user] = list(set(user_itemlist).difference(set(self.userDict[user][0])))

            # Sample for train set
            num = 0
            while num < NEGATIVE_NUM and user_itemlist != []:
                negativeItemid = user_itemlist.pop()
                if not negativeItemid in self.userDict[user][0]:
                    self.train_memory_neg.append((user, memories, negativeItemid, 0))
                    num += 1

        # Supply the negative items to test set
        for user, memories, itemid, rating in self.test_memory_pos:
            try:
                neg = test_neg_dic[user]
                self.test_memory_neg.append((user, memories, negativeItemid, 0))
            except:
                continue

        '''
        # This method takes too long time!
        negative_exist = {}
        for user, memories, itemid, rating in self.test_memory_pos:
            # only one test item for each user, so we can set default each iteration
            # when more than one item in test set, this should be modified
            negative_exist.setdefault(user, [])
            for j in range(NEGATIVE_NUM):
                while 1:
                    negativeItem = random.randint(0, self.itemNum-1)
                    negativeItemid = self.itemList[negativeItem][0]
                    if negativeItemid not in self.userDict[user][0] + negative_exist[user]:
                        self.test_memory_neg.append((user, memories, negativeItemid, 0))
                        break
        for user, memories, itemid, rating in self.train_memory_pos:
            for i in range(NEGATIVE_NUM):
                while 1:
                    negativeItem = random.randint(0, self.itemNum-1)
                    negativeItemid = self.itemList[negativeItem][0]
                    if negativeItemid not in self.userDict[user][0] + negative_exist[user]:
                        self.train_memory_neg.append((user, memories, negativeItemid, 0))
                        negative_exist[user].append(negativeItemid)
                        break
        '''

        t2 = time.time()
        print('Negative sampling done! Time: %.2fms' % ((t2-t1) * 1000), file=sys.stderr)

    def memory_merge_pos_neg(self):
        self.train_memory_merge = self.train_memory_pos + self.train_memory_neg
        self.test_memory_merge = self.test_memory_pos + self.test_memory_neg
        random.shuffle(self.train_memory_merge)
        random.shuffle(self.test_memory_merge)

    def init_memory(self, NEGATIVE_NUM=4):
        self.memory_orderByTime()
        self.memory_generate_train_withOneItemOut()
        self.memory_negative_sampling(NEGATIVE_NUM)
        self.memory_merge_pos_neg()

    '''
    SAVE GENERATED MEMORY DATA TO LOCAL FOR REPEATABLE USAGE
    ==============================================
    '''
    def init_path(self):
        global file_paths, file_memory_paths
        global csv_paths, csv_memory_paths
        global gen_data, load_data
        global gen_memory_data, load_memory_data

        # txt file path
        training_memory_path = os.path.join(self.root, 'ml-1m.train.merge.memory.txt')
        test_memory_pos_path = os.path.join(self.root, 'ml-1m.test.pos.memory.txt')
        test_memory_neg_path = os.path.join(self.root, 'ml-1m.test.neg.memory.txt')
        file_memory_paths = [training_memory_path, test_memory_pos_path, test_memory_neg_path]

        training_path = os.path.join(self.root, 'ml-1m.train.merge.txt')
        test_pos_path = os.path.join(self.root, 'ml-1m.test.pos.txt')
        # test_neg = os.path.join(self.root, 'ml-1m.test.neg.txt')
        file_paths = [training_path, test_pos_path]

        # csv file path
        training_memory_path = os.path.join(self.root, 'ml-1m.train.merge.memory.csv')
        test_memory_pos_path = os.path.join(self.root, 'ml-1m.test.pos.memory.csv')
        test_memory_neg_path = os.path.join(self.root, 'ml-1m.test.neg.memory.csv')
        csv_memory_paths = [training_memory_path, test_memory_pos_path, test_memory_neg_path]

        training_path = os.path.join(self.root, 'ml-1m.train.merge.csv')
        test_pos_path = os.path.join(self.root, 'ml-1m.test.pos.csv')
        # test_neg = os.path.join(self.root, 'ml-1m.test.neg.txt')
        csv_paths = [training_path, test_pos_path]

        # data to save
        gen_memory_data = [self.train_memory_merge, self.test_memory_pos, self.test_memory_neg]
        gen_data = [self.trainList, self.testList]

        # data to load
        train_memory_merge, test_memory_pos, test_memory_neg = [], [], []
        load_memory_data = [train_memory_merge, test_memory_pos, test_memory_neg]
        train_merge, test_pos = [], []
        load_data = [train_merge, test_pos]

    def save_to_txt(self, SAVE_MEMORY=True):
        # When saving to file, each line must be string
        if SAVE_MEMORY:
            for i in range(3):
                with open(file_memory_paths[i], 'w') as f:
                    for line in gen_memory_data[i]:
                        f.write(str(line))
        else:
            for i in range(2):
                with open(file_paths[i], 'w') as f:
                    for line in gen_data[i]:
                        f.write(str(line))
            print('Save fixed memory dataset as file success!', file=sys.stderr)

    def save_to_csv(self, SAVE_MEMORY=True):
        import pandas as pd
        if SAVE_MEMORY:
            for i in range(3):
                df = pd.DataFrame(data=gen_memory_data[i], columns=None)
                df.to_csv(csv_memory_paths[i], index=None, columns=None)
                print('Save fixed memory dataset %s to csv success!' % csv_memory_paths[i], file=sys.stderr)
                # col1, col2, col3, col4 = [], [], [], []
                # for user, history, item, rating in gen_data[i]:
                #     col1.append(user)
                #     col2.append(history)
                #     col3.append(item)
                #     col4.append(rating)
        else:
            for i in range(2):
                df = pd.DataFrame(data=gen_data[i], columns=None)
                df.to_csv(csv_paths[i], index=None, columns=None)

    def load_generated_memory_txt(self):
        '''before loading data remember to initialize the paths'''
        t1 = time.time()
        for i in range(3):
            with open(file_memory_paths[i], 'w') as f:
                line = f.readlines()
                load_memory_data[i].append(line)
        # train_memory_merge, test_memory_pos, test_memory_neg = load_memory_data
        t2 = time.time()
        print('Load fixed memory dataset as lines success! Time: %.2fms' % ((t2-t1)*1000), file=sys.stderr)
        return load_memory_data

    def load_generated_memory_csv(self):
        t1 = time.time()
        import pandas as pd
        train_memory_merge_tf = pd.read_csv(csv_memory_paths[0])
        test_memory_pos_tf = pd.read_csv(csv_memory_paths[1])
        test_memory_neg_tf = pd.read_csv(csv_memory_paths[2])
        t2 = time.time()
        print('Load fixed memory dataset as Dataframe success! Time: %.2fms' % ((t2-t1)*1000), file=sys.stderr)
        return train_memory_merge_tf, test_memory_pos_tf, test_memory_neg_tf


def preprocess_data(ratings, users, movies):
    ## Process user data
    # filter and delete lists with null elements
    users = users.filter(regex='UserID|Gender|Age|OccupationID')  # Zip-code
    users_orig = users.values
    # map Gender and Age to number
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    ## Process movie data
    movies_orig = movies.values
    # remove year from movie Title
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    # movie Genre to number/index dict
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # Turn movie Genre into equal-length number list, with length 18 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

    movies['Genres'] = movies['Genres'].map(genres_map)

    # movie Title to number/index dict
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # Turn movie Genre into equal-length number list, with length 15
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

    movies['Title'] = movies['Title'].map(title_map)

    # delete lists with null elements in ratings
    ratings = ratings.filter(regex='UserID|MovieID|Rating')  # timestamps

    # merge dataframes
    data = pd.merge(pd.merge(ratings, users), movies)

    # divide data into two tables: x and y
    target_fields = ['Rating']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


'''
DETAILED DESCRIPTIONS OF DATA FILES
==============================================
Here are brief descriptions of the data.
ml-data.tar.gz   -- Compressed tar file.  To rebuild the u data files do this:
                gunzip ml-data.tar.gz
                tar xvf ml-data.tar
                mku.sh
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
              user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   
u.info     -- The number of users, items, and ratings in the u data set.
u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.
u.genre    -- A list of the genres.
u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set.
u.occupation -- A list of the occupations.
u1.base    -- The data sets u1.base and u1.test through u5.base and u5.test
u1.test       are 80%/20% splits of the u data into training and test data.
u2.base       Each of u1, ..., u5 have disjoint test sets; this if for
u2.test       5 fold cross validation (where you repeat your experiment
u3.base       with each training and test set and average the results).
u3.test       These data sets can be generated from u.data by mku.sh.
u4.base
u4.test
u5.base
u5.test
ua.base    -- The data sets ua.base, ua.test, ub.base, and ub.test
ua.test       split the u data into a training set and a test set with
ub.base       exactly 10 ratings per user in the test set.  The sets
ub.test       ua.test and ub.test are disjoint.  These data sets can
              be generated from u.data by mku.sh.
allbut.pl  -- The script that generates training and test sets where
              all but n of a users ratings are in the training data.
mku.sh     -- A shell script to generate all the u data sets from u.data.
'''


class Movielens_100k:
    def __init__(self, root='./Data/ml-100k/', train=None, test=None):
        self.root = root
        self.ratingList = []
        self.userList = []
        self.itemList = []
        self.train = train
        self.test = test
        self.trainList = []
        self.testList = []
        self.trainDict = {}
        self.testDict = {}
        self.genre = {}
        self.occupation = {}

    def load_training_test(self, fileName='u1'):
        t1 = time.time()
        root_training = os.path.join(self.root, fileName + '.base')
        root_test = os.path.join(self.root, fileName + '.test')
        file_training = open(root_training, 'r')
        file_test = open(root_test, 'r')

        for line1 in file_training:
            userid, itemid, rating, timestamp = line1.split('\t')
            self.trainList.append([int(userid), int(itemid), float(rating), int(timestamp)])
        for line2 in file_test:
            userid, itemid, rating, timestamp = line2.split('\t')
            self.testList.append([int(userid), int(itemid), float(rating), int(timestamp)])
        file_training.close()
        file_test.close()
        t2 = time.time()
        print('Load %s data success! Training data: %d,Test data:%d,Time: %.2fms' % (
            fileName, len(self.trainList), len(self.trainList), (t2 - t1) * 1000), file=sys.stderr)

    def load_user_data(self):
        t1 = time.time()
        path = os.path.join(self.root, 'u.user')
        fp = open(path, 'r')
        for line in fp:
            uid, age, gender, occupation, zipcode = line.strip('\n').split('|')
            self.userList.append([int(uid), gender, int(age), occupation, zipcode])
        fp.close()
        t2 = time.time()
        print('Load user data success! Total users: %d,Time: %.2fms' % (len(self.userList), (t2 - t1) * 1000),
              file=sys.stderr)

    def load_item_data(self):
        t1 = time.time()
        path = os.path.join(self.root, 'u.item')
        fp = open(path, 'r', encoding='latin1')
        for line in fp:
            iid, title, releaseDate, videoReleaseDate, IMDb_URL, \
            unknown, Action, Adventure, Animation, Children, Comedy, \
            Crime, Documentary, Drama, Fantasy, FilmNoir, Horror, \
            Musical, Mystery, Romance, SciFi, Thriller, War, Western = line.strip('\n').split('|')
            self.itemList.append([int(iid), title, releaseDate, videoReleaseDate])
        fp.close()
        t2 = time.time()
        print('Load item data success! Total items: %d,Time: %.2fms' % (len(self.itemList), (t2 - t1) * 1000),
              file=sys.stderr)

    def load_dict(self):
        t1 = time.time()
        path_genre = os.path.join(self.root, 'u.genre')
        path_occupation = os.path.join(self.root, 'u.occupation')
        file_genre = open(path_genre, 'r')
        file_occupation = open(path_occupation, 'r')
        for line in file_genre:
            gen, id = line.strip('\n').split('|')
            self.genre[gen] = id
        for i, occ in enumerate(file_occupation):
            self.occupation[occ] = i
        file_genre.close()
        file_occupation.close()
        t2 = time.time()
        print('Load genre and occupation dictionaries success! Time: %.2fms' % ((t2 - t1) * 1000), file=sys.stderr)

    def getDict(self):
        for line in self.trainList:
            userid, itemid, record, _ = line
            self.trainDict.setdefault(userid, {})
            self.trainDict[userid][itemid] = record
        for line in self.testList:
            userid, itemid, record, _ = line
            self.testDict.setdefault(userid, {})
            self.testDict[userid][itemid] = record

    def getDataframe(self):
        self.load_training_test()
        self.load_user_data()
        self.load_item_data()
        rating_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
        user_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
        item_title = ['MovieID', 'Title', 'ReleaseDate', 'VideoReleaseDate']
        ratings = pd.DataFrame(data=self.trainList, columns=rating_title)
        users = pd.DataFrame(data=self.userList, columns=user_title)
        items = pd.DataFrame(data=self.itemList, columns=item_title)
        return ratings, users, items


if __name__ == "__main__":
    data = Movielens_1m('Data\ml-1m')
    data.init_memory()
    data.init_path()
    data.save_to_csv(SAVE_MEMORY=True)
    # train, test_pos, test_neg = data.load_generated_memory_csv()

    # gender_map = {'F':0, 'M':1}
    # users['Gender'] = users['Gender'].map(gender_map)
    # print(users)
