# from Supervised_method_KP_Extraction.read_train_test_data import *
# from Supervised_method_KP_Extraction.extract_candidate_chunks_and_words import *
# path = "./SemEval2010"
# train_docs, train_labels = reader(path)
#
# doc = train_docs["C-41.txt.final"]
# doc = " ".join(doc)
# label = train_labels[0]
# label = label.replace("C-41 : ", "").split(",")
# candidates = extract_candidate_chunks(doc)


# ===== test for is_labeled_keyPhrases =============
# def is_labeled_keyPhrases(candidate, doc_labeled_keyPhrases):
#     import nltk
#     """
#     check if the candidate belong to labeled key phrases
#     :param candidate:
#     :param doc_labeled_keyPhrases:
#     :return: yes 1; no 0
#     """
#     # TODO: Because labeled key phrase are Stemming form, So firstly change candidate in stemming form
#     from nltk.stem.porter import PorterStemmer
#     flag = False
#     stemmer = PorterStemmer()
#     candidate_in_words = [stemmer.stem(word) for word in nltk.word_tokenize(candidate)]
#     stem_candidate = " ".join(candidate_in_words)
#     # stem_candidate = " ".join([stemmer.stem(word) for word in candidate])
#     if stem_candidate in doc_labeled_keyPhrases:
#         flag = True
#     # if candidate is included in any labeled key phrase
#     # if flag is False:
#     #     for labeled_cand in doc_labeled_keyPhrases:
#     #         if stem_candidate in labeled_cand:
#     #             flag = True
#     #             break
#
#     # if candidate include  any labeled key phrase
#     if flag is False:
#         for labeled_cand in doc_labeled_keyPhrases:
#             if labeled_cand in stem_candidate:
#                 flag = True
#                 break
#     if flag is True:
#         return 1
#     else:
#         return 0

# y = []
# for cand in candidates:
#     is_label = is_labeled_keyPhrases(cand, label)
#     y.append(is_label)
# print(candidates)


# =============== test for train_X_1201, train_y_1201 ====================
import pickle
train_y = pickle.load(open("train_y_1201.p", "rb"))
from collections import Counter
print(Counter(train_y[:20000]))
