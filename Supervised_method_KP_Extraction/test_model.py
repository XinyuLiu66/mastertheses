from Supervised_method_KP_Extraction.train_classifier import *
# from Supervised_method_KP_Extraction.data_preprocessing import *
from Supervised_method_KP_Extraction.read_train_test_data import *
import numpy as np

# title = "intro to Automatic Keyphrase Extraction"
# text = "I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains."
# candidates = extract_candidate_chunks(text)

# =================== predict key phrases of given document
def predict_key_phrases(rank_svm, doc, title=False, abstract=False):
    """
    predict key phrases of given document
    :param doc: a doc, list of line
    :return: key phrases of this document
    """
    if title:
        title = title
    else:
        title = None
    if abstract:
        abstract == abstract
    else:
        abstract = None
    candidates = get_candidates(doc)
    print("Candidates \n", candidates)
    doc_text = " ".join(doc)

    # print("==== title =======")
    # print(title)
    # print("==== abstract =======")
    # print(abstract)
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

# import pickle
clf = pickle.load(open("rank_svm_1201_10000.p", "rb"))
# _, test_docs, _ = reader("./SemEval2010")
# test_text = test_docs["C-1.txt.final"]
#
#
# train_X = pickle.load(open("train_X_1130.p", "rb"))
# train_y = pickle.load(open("train_y_1130.p", "rb"))
# print("==============")
# #print("score : ", clf.score(train_X[:10000], train_y[:10000]))
# from collections import Counter
# print(Counter(train_y[:10000]))

doc_text = "I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains." \
           "Despite wide applicability and much research, keyphrase extraction suffers from poor performance relative to many other core NLP tasks, partly because there’s no objectively “correct” set of keyphrases for a given document. While human-labeled keyphrases are generally considered to be the gold standard, humans disagree about what that standard is! As a general rule of thumb, keyphrases should be relevant to one or more of a document’s major topics, and the set of keyphrases describing a document should provide good coverage of all major topics. (They should also be understandable and grammatical, of course.) The fundamental difficulty lies in determining which keyphrases are the most relevant and provide the best coverage. As described in Automatic Keyphrase Extraction: A Survey of the State of the Art, several factors contribute to this difficulty, including document length, structural inconsistency, changes in topic, and (a lack of) correlations between topics."
doc_title = "Intro to Automatic Keyphrase Extraction"
import nltk
doc_text = nltk.sent_tokenize(doc_text)
# print(test_text)
kp = predict_key_phrases(clf, doc_text, title=doc_title)
print(kp)

# import pickle
# from collections import Counter
# train_y = pickle.load(open("train_y_1201.p", "rb"))
# print(Counter(train_y))