import re
from Supervised_method_KP_Extraction.extract_candidate_chunks_and_words import *
from Supervised_method_KP_Extraction.feature_engineering import *
import numpy as np
import collections

# demo, only read one file, later will read a folder
def reader(path):
    with open(path,'r') as file:
        doc = []
        for line in file:
            doc.append(line.replace("\n",""))
    return doc


def get_title_abs_text(doc):
    """
    input:
        doc: a list of line from reader
    :return: string
        title:
        abstract:
        text:
    """
    title_index = doc.index('TITLE')
    author_index = doc.index('AUTHOR')
    abstract_index = doc.index('ABSTRACT')
    text_index = doc.index('1. INTRODUCTION')
    title = " ".join(doc[title_index+1: author_index])
    abstract = " ".join(doc[abstract_index+1: text_index])
    text = " ".join(doc[text_index: ])
    return title, abstract, text

def get_candidates(text):
    candidates = extract_candidate_chunks(text)
    return candidates

def prepare_parameter_for_feature_fun(path):
    doc = reader(path)
    text = " ".join(doc)
    candidates = get_candidates(text)
    title, abstract, text = get_title_abs_text(doc)
    return candidates,title,abstract,text


def get_label_candidate(path):
    label_cand = []
    with open(path) as file:
        for line in file:
            if line.startswith("C-41"):
                line = line.replace("C-41 : ","")
                line = line.replace("\n", '')
                label_cand.append(line)
                break
    labeled_cand = label_cand[0].split(",")
    return labeled_cand


def get_final_traindata(test_path, labeled_path):

    candidates, doc_title, doc_abstract, doc_text = prepare_parameter_for_feature_fun(test_path)
    labeled_candidates = get_label_candidate(labeled_path)
    labeled_candidates_by_words = []
    for cand in labeled_candidates:
        labeled_candidates_by_words.extend(cand.split(" "))
    candidate_scores = extract_candidate_features(candidates, doc_text, doc_abstract, doc_title)
    tuple_candidate_score = list(candidate_scores.items())

    # fortest
    # print("==============")
    # print(tuple_candidate_score[0])

    train_X = []
    train_y = []
    for key, features_score in tuple_candidate_score:
        feature_set = []
        feature_set.append(features_score['term_count'])
        feature_set.append(features_score['term_length'])
        feature_set.append(features_score['max_word_length'])
        feature_set.append(features_score['spread'])
        feature_set.append(features_score['lexical_cohesion'])
        feature_set.append(features_score['in_excerpt'])
        feature_set.append(features_score['in_title'])
        feature_set.append(features_score['abs_first_occurrence'])
        feature_set.append(features_score['abs_last_occurrence'])
        #TODO  a little bit change
        if(set(key.split(" ")).intersection(set(labeled_candidates_by_words))):
            train_y.append(1)
        else:
            train_y.append(0)
        train_X.append(feature_set)
    train_X = np.array(train_X)
    return train_X, train_y





# # # test reader
text_path = "./C-41.txt.final"
# doc = reader(path)
# #print(doc)
# candidates,_,_,_ = prepare_parameter_for_feature_fun(path)
# print(candidates)
#


# test label
label_path = "./train.combined.final"
# label_cand = get_label_candidate(path)
# print(label_cand)

train_datasets = get_final_traindata(text_path, label_path)
