
import pickle
import sys
sys.path.append('../')
from Supervised_method_KP_Extraction.rankSVM import RankSVM
# from Supervised_method_KP_Extraction.read_train_test_data import *


train_X = pickle.load(open("train_X_1201.p", "rb"))
train_y = pickle.load(open("train_y_1201.p", "rb"))


# ============== train classifier ============== #

rank_svm = RankSVM().fit(train_X[:2000], train_y[:2000])
pickle.dump(rank_svm, open("rank_svm_1201_10.p", "wb"))



#import pickle
#classifier = pickle.dump(rank_svm, open("rank_svm_1130.p", "wb"))
# train_X = train_X[5000:10000]
# train_y = train_y[5000:10000]
# print(classifier.coef_)

