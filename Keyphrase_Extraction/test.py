import numpy as np
#
# np.random.seed(0)
# X = np.random.randn(6,2)
# # print("Before X = ", X)
#
# X[::1] -= np.array([3, 7])
# # print("After X = ", X)
#
# blocks = np.array([0, 1] * int(X.shape[0] / 2))
# print(blocks)



# from sklearn.model_selection import StratifiedShuffleSplit
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1])
# sss = StratifiedShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
#
# for train_index, test_index in sss.split(X, y):
#     print("TRAIN Index:", train_index, "TEST Index:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     print("TRAIN Value: ", X_train, "TEST Value", X_test)
#     y_train, y_test = y[train_index], y[test_index]
# #
# from sklearn import svm, linear_model, cross_validation
#
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[7,8],[8,9]])
# y = np.array([1, 2, 3, 4])
# kf = cross_validation.KFold(6,3)
#
# train, test = iter(kf).__next__()
#
# print("train: ", train)
# print("test: ", test)


from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

# X, y = make_classification(n_features=4, random_state=0)
# print("X: ",X.shape)
# print("y: ", y.shape)
# clf = LinearSVC(random_state=0)
# clf.fit(X,y)
# print(clf.coef_)
# import nltk
# text = "I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains."
#
# tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent)
#                                       for sent in nltk.sent_tokenize(text))
# import pickle
# pickle.dump(tagged_sents, open("save.p", "wb"))
#
# x = pickle.load(open("save.p", "rb"))
#
# print(x)

import numpy as np
a = [[1,2],[2,3],[3,5]]
n_a = np.array(a)
print(n_a[:2])