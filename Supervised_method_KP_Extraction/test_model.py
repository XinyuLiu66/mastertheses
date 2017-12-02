import numpy as np
import sys
import pickle
import os
sys.path.append('../')
# from Supervised_method_KP_Extraction.read_train_test_data import get_candidates, get_features_of_candidate
from Supervised_method_KP_Extraction.feature_engineering import extract_candidate_features
from Supervised_method_KP_Extraction.extract_candidate_chunks_and_words import *
# import read_train_test_data



# =================== predict key phrases of given document
def predict_key_phrases(rank_svm, doc):
    """
    predict key phrases of given document
    :param doc: a doc, list of line
    :return: key phrases of this document
    """
    title = get_title(doc)
    if len(title) == 0:
        title == None

    abstract = get_abstract(doc)
    if len(abstract) == 0:
        abstract = None

    candidates = get_candidates(doc)
    print("Candidates \n", candidates)
    doc_text = " ".join(doc)

    dic_candidates_features = extract_candidate_features(candidates,
                                                         doc_text,abstract,title)
    # make a candidates dictionary. e.g. dic{num: key phrase}
    # to get candidate given the index
    dic_candidates = {}
    list_candidates = []
    for cand in dic_candidates_features.keys():
        list_candidates.append(cand)
    for i, cand in enumerate(list_candidates):
        dic_candidates[i] = cand
    # get features set according to the order of candidate in list_candidates
    test_X = get_input_data(list_candidates, dic_candidates_features)
    rank = rank_svm.predict(test_X)

    # get key phrases according to the rank
    keyphrases = []
    for i in rank:
        keyphrases.append(dic_candidates[i])
    return keyphrases




# ======= called by predict_key_phrases =======
def get_features_of_candidate(candidate, dic_candidates_feature_set):
    """
    get the features of each candidate
    :param candidate: a candidate in doc
    :param dic_candidates_feature_set: the features of all candidates in one documents
    :return: features of this doc, in list
    """
    cand_features = []
    cand_features.append(dic_candidates_feature_set[candidate]['term_count'])
    cand_features.append(dic_candidates_feature_set[candidate]['term_length'])
    cand_features.append(dic_candidates_feature_set[candidate]['max_word_length'])
    cand_features.append(dic_candidates_feature_set[candidate]['spread'])
    cand_features.append(dic_candidates_feature_set[candidate]['lexical_cohesion'])
    cand_features.append(dic_candidates_feature_set[candidate]['in_excerpt'])
    cand_features.append(dic_candidates_feature_set[candidate]['in_title'])
    cand_features.append(dic_candidates_feature_set[candidate]['abs_first_occurrence'])
    cand_features.append(dic_candidates_feature_set[candidate]['abs_last_occurrence'])
    return cand_features

# ======= called by predict_key_phrases =======
def get_candidates(doc):
    """
    This method is called by data pre-processing return candidates of a doc
    :param doc: a list of line
    :return: doc_candidates
    """
    text = " ".join(doc)
    doc_candidates = extract_candidate_chunks(text)
    return doc_candidates

# ======= called by predict_key_phrases =======
def get_input_data(list_candidates, dic_candidates_features):
    """
    get features set according to the order of candidate in list_candidates
    :param list_candidates:
    :param dic_candidates_features:
    :return: X
    """
    X = []
    for cand in list_candidates:
        cand_features = get_features_of_candidate(cand,dic_candidates_features)
        X.append(cand_features)
    X = np.array(X)
    return X


# ===== call by data pre-processing =============
def get_title(doc):
    """
    This method is called by data pre-processing return title of a doc
    :param doc: a list of line
    :return: doc_title
    """
    title_list = []
    for line in doc:
        if line.startswith("ABSTRACT"):
            break
        title_list.append(line)
    title = " ".join(title_list)
    return title

# ===== call by data pre-processing =============
def get_abstract(doc):
    """
    This method is called by data pre-processing return abstract of a doc
    :param doc: a list of line
    :return: doc_abstract
    """
    flag = False
    abstract_list = []
    for line in doc:
        if line.startswith("ABSTRACT"):
            flag = True
        if line.startswith("1. INTRODUCTION"):
            flag = False
            break
        if flag is True:
            abstract_list.append(line)

    abstract = " ".join(abstract_list[1:])
    return abstract


# ============================= Part 1 =====================================
def reader(path):
    """
    This method read train and test documents
    :param path: path of documents
    :return: train_docs, label file for each train doc, test_docs
     Dictionary
                e.g ["file_name": document in list]
    """

    # ===== get train and test documents ========== #
    train_data_path = path + "/train"
    test_data_path = path + "/test"

    train_docs = get_docs(train_data_path)
    test_docs = get_docs(test_data_path)


    # read labeled file for train documents
    label_file = train_data_path + "/train.combined.stem.final"
    labeled_keyPhrases = []
    with open(label_file, "rt") as lf:
        for line in lf:
            labeled_keyPhrases.append(line.replace("\n",""))

    return train_docs, test_docs,  labeled_keyPhrases




# =================== call by reader =============
def get_docs(dir_path):
    """
    This methos is called by reader
    :param path:  file path
    :return: list of dictionary
                e.g ["file_name": document]
    """
    docs = dict()
    files = os.listdir(dir_path)
    files = [file for file in files if file.endswith("txt.final")]
    for file in files:
        if (os.path.isdir(file)):
            continue
        else:
            f = open(dir_path + "/" + file, "rt")
            text = []
            for line in f:
                text.append(line.replace("\n", ""))
            docs[file] = text
    return docs

# ============================================================================================== #



# import pickle
clf = pickle.load(open("rank_svm_1202_4000.p", "rb"))
print(clf.coef_)
# train_docs, test_docs, _ = reader("./SemEval2010")
# doc_text = train_docs["C-41.txt.final"]
#
#
# train_X = pickle.load(open("train_X_1130.p", "rb"))
# train_y = pickle.load(open("train_y_1130.p", "rb"))
# print("==============")
# #print("score : ", clf.score(train_X[:10000], train_y[:10000]))
# from collections import Counter
# print(Counter(train_y[:10000]))

# doc_text = "I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains." \
#            "Despite wide applicability and much research, keyphrase extraction suffers from poor performance relative to many other core NLP tasks, partly because there’s no objectively “correct” set of keyphrases for a given document. While human-labeled keyphrases are generally considered to be the gold standard, humans disagree about what that standard is! As a general rule of thumb, keyphrases should be relevant to one or more of a document’s major topics, and the set of keyphrases describing a document should provide good coverage of all major topics. (They should also be understandable and grammatical, of course.) The fundamental difficulty lies in determining which keyphrases are the most relevant and provide the best coverage. As described in Automatic Keyphrase Extraction: A Survey of the State of the Art, several factors contribute to this difficulty, including document length, structural inconsistency, changes in topic, and (a lack of) correlations between topics."
# doc_title = "Intro to Automatic Keyphrase Extraction"
# import nltk
# doc_text = nltk.sent_tokenize(doc_text)
# print(test_text)


# kp = list(reversed(predict_key_phrases(clf, doc_text)))
# print(" ==== KP ======")
# for i,x in enumerate(kp):
#     print(i, x)
#     if i == 20:
#         break
# print(kp[:20])

