import os
from Supervised_method_KP_Extraction.extract_candidate_chunks_and_words import *
from Supervised_method_KP_Extraction.feature_engineering import *

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

    return train_docs,  test_docs, labeled_keyPhrases


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

# =======  demo test =============
# path = "./SemEval2010"
# train_docs, test_docs, label_file = reader(path)
# print(train_docs["C-41.txt.final"])


# ============================= Part 2 =====================================
# =======data pre-processing for train data ============
def data_preprocessing_train(docs, labeled_keyPhrases):
    """
    This method change the text data to training data sets
    :param docs: a dictionary, key:file name ; value: text in the list form
           labeled_keyPhrases: a list, each ele include labeled key phrase of
                                each document, *** in stemming form ***
    :return: train_X, train_y
             train_X : candidates' feature data shape(n_sample, n_feature)
             train_y : label if the candidate is key phrase, yes is 1, no is 0
                        shape(n_sample)
    """
    import numpy as np
    candidates_feature_set = []
    labels = []
    for file_name, doc in docs.items():
        doc_title = get_title(doc)
        doc_abstract = get_abstract(doc)
        doc_text = " ".join(doc)
        doc_candidates = get_candidates(doc)
        doc_labeled_keyPhrases = get_labeled_keyphrases(file_name, labeled_keyPhrases)

        # get features sets of the candidates
        dic_candidates_feature_set = extract_candidate_features(doc_candidates,
                                                          doc_text, doc_abstract, doc_title)

        for cand in dic_candidates_feature_set.keys():
            cand_features = get_features_of_candidate(cand, dic_candidates_feature_set)
            candidates_feature_set.append(cand_features)
            # check if the candidate belong to labeled key phrases
            labels.append(is_labeled_keyPhrases(cand,doc_labeled_keyPhrases))

    train_X = np.array(candidates_feature_set)
    train_y = np.array(labels)
    return train_X,train_y


# =======data pre-processing for test data ============
def data_preprocessing_test(docs):
    """
    This method change the test text data to features set
    :param docs: a dictionary, key:file name ; value: text in the list form
    :return: test_X
             test_X : candidates' feature data shape(n_sample, n_feature)

    """
    import numpy as np
    candidates_feature_set = []
    for file_name, doc in docs.items():
        doc_title = get_title(doc)
        doc_abstract = get_abstract(doc)
        doc_text = " ".join(doc)
        doc_candidates = get_candidates(doc)

        # get features sets of the candidates
        dic_candidates_feature_set = extract_candidate_features(doc_candidates,
                                                          doc_text, doc_abstract, doc_title)
        for cand in dic_candidates_feature_set.keys():
            cand_features = get_features_of_candidate(cand, dic_candidates_feature_set)
            candidates_feature_set.append(cand_features)

    test_X = np.array(candidates_feature_set)
    return test_X


# ===== call by data pre-processing =============
def is_labeled_keyPhrases(candidate, doc_labeled_keyPhrases):
    import nltk
    """
    check if the candidate belong to labeled key phrases
    :param candidate:
    :param doc_labeled_keyPhrases:
    :return: yes 1; no 0
    """
    # TODO: Because labeled key phrase are Stemming form, So firstly change candidate in stemming form
    from nltk.stem.porter import PorterStemmer
    flag = False
    stemmer = PorterStemmer()
    candidate_in_words = [stemmer.stem(word) for word in nltk.word_tokenize(candidate)]
    stem_candidate = " ".join(candidate_in_words)
    # stem_candidate = " ".join([stemmer.stem(word) for word in candidate])
    if stem_candidate in doc_labeled_keyPhrases:
        flag = True
    # if candidate is included in any labeled key phrase
    # if flag is False:
    #     for labeled_cand in doc_labeled_keyPhrases:
    #         if stem_candidate in labeled_cand:
    #             flag = True
    #             break

    # if candidate include  any labeled key phrase
    if flag is False:
        for labeled_cand in doc_labeled_keyPhrases:
            if labeled_cand in stem_candidate:
                flag = True
                break
    if flag is True:
        return 1
    else:
        return 0


# ===== call by data pre-processing =============
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


# demo test get_title and get_abstract method
    # doc = train_docs["C-41.txt.final"]
    # title = get_title(doc)
    # abstract = get_abstract(doc)
    # print("Title: \n", title)
    # print("Abstract: \n", abstract)

# ===== call by data pre-processing =============
def get_candidates(doc):
    """
    This method is called by data pre-processing return candidates of a doc
    :param doc: a list of line
    :return: doc_candidates
    """
    text = " ".join(doc)
    doc_candidates = extract_candidate_chunks(text)
    return doc_candidates

# =================== call by data pre-processing =============
def get_labeled_keyphrases(file_name, labeled_keyPhrases):
    """
    This method is called by data pre-processing return labeled key phrases of a doc
    :param doc: a list of line
    :return: doc_labeled_keyPhrases
    """
    key_phrases = []
    file_name = file_name.replace(".txt.final", "")
    for line in labeled_keyPhrases:
        if line.startswith(file_name):
            line = line.replace(file_name + " : ", "")
            key_phrases = line.split(",")
            break
    return key_phrases

# demo test get_candidates and get_labeled_keyphrases
    # doc_candidates = get_candidates(doc)
    # print("doc_candidates: \n", doc_candidates)
    # key_phrases = get_labeled_keyphrases("C-41.txt.final", label_file)
    # print("key phrases: \n", key_phrases)


# ============== demo for all the methods ==================

path = "./SemEval2010"
train_docs, label_file = reader(path)
train_X, train_y = data_preprocessing_train(train_docs, label_file)


import pickle
pickle.dump(train_X, open("train_X_1201.p", "wb"))
pickle.dump(train_y, open("train_y_1201.p", "wb"))




