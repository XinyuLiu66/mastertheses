from New_KeyPhrase_Exraction.extract_candidate_chunks_and_words import *


# =============score kandidate by tf*idf=================
def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    import gensim
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]

    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]

    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf, dictionary





# score each candidate using titles and keywords information, give bias to each candidate
def get_titles_keywords_by_words(titles, keywords):
    titles_by_words = list()
    keywords_by_words = list()
    for phrase in titles:
        titles_by_words.extend(phrase.split())
    for keyword in keywords:
        keywords_by_words.extend(keyword.split())
    return titles_by_words, keywords_by_words


def scroe_again_to_candidate(all_chunks, dictionary, keywords, titles):
    BIAS_TO_WORD_IN_TITLE = 2
    BIAS_TO_WORD_IN_KEYWORD = 4
    for i, chunk in enumerate(all_chunks):
        if set(dictionary[chunk[0]].split(" ")).intersection(set(keywords)):
            chunk = list(chunk)
            chunk[1] *= BIAS_TO_WORD_IN_KEYWORD
            chunk = tuple(chunk)
            all_chunks[i] = chunk
        elif set(dictionary[chunk[0]].split(" ")).intersection(set(titles)):
            chunk = list(chunk)
            chunk[1] *= BIAS_TO_WORD_IN_TITLE
            chunk = tuple(chunk)
            all_chunks[i] = chunk
    return all_chunks




            # ===== test  score_keyphrases_by_tfidf() =======
# corpus_tfidf, dictionary =  score_keyphrases_by_tfidf(texts)
# results = []
# for doc in corpus_tfidf:
#     for key, tfidf in doc:
#         results.append((tfidf, dictionary[key]))
