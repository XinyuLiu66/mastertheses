from Supervised_method_KP_Extraction.rankSVM import *
from Supervised_method_KP_Extraction.read_train_test_data import *
from Supervised_method_KP_Extraction.test_model import *
import pickle

train_X = pickle.load(open("train_X.p", "rb"))
train_y = pickle.load(open("train_y.p", "rb"))
test_X = pickle.load(open("test_X.p", "rb"))

# ============== train classifier ============== #

rank_svm = RankSVM()
rank_svm.fit(train_X, train_y)
# pickle.dump(rank_svm, open("test_rank_svm.p", "wb"))



import pickle
classifier = pickle.load(open("test_rank_svm.p", "rb"))
# train_X = train_X[5000:10000]
# train_y = train_y[5000:10000]
# print(classifier.coef_)

