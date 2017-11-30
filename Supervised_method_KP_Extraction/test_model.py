from Supervised_method_KP_Extraction.train_classifier import *
from Supervised_method_KP_Extraction.data_preprocessing import *
from Supervised_method_KP_Extraction.read_train_test_data import *
import numpy as np

# title = "intro to Automatic Keyphrase Extraction"
# text = "I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains."
# candidates = extract_candidate_chunks(text)

# =================== predict key phrases of given document
def predict_key_phrases(rank_svm, doc):
    """
    predict key phrases of given document
    :param doc: a doc, list of line
    :return: key phrases of this document
    """
    title = get_title(doc)
    abstract = get_abstract(doc)
    candidates = get_candidates(doc)
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
def get_input_data(list_candidates, dic_candidates_features):
    """
    get features set according to the order of candidate in list_candidates
    :param list_candidates:
    :param dic_candidates_features:
    :return: X
    """
    X = []
    for cand in list_candidates:
        pass
      #  cand_features = get_features_of_candidate(cand,dic_candidates_features)
      #   X.append(cand_features)
    X = np.array(X)
    return X

# import pickle
classifier = pickle.load(open("test_rank_svm.p", "rb"))
_, test_docs, _ = reader("./SemEval2010")
test_text = test_docs["C-1.txt.final"]

kp = predict_key_phrases(rank_svm, test_text)
print(kp)
