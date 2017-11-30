
# ============test =================
text1 = "Deep neural networks are accurate predictors, but their deci- sions are difficult to interpret, which limits their applicability in various fields. " \
       "Symbolic representations in the form of rule sets are one way to illustrate their behavior as a whole, as well as the hidden concepts they model in the intermediate layers. " \
       "The main contribution of the paper is to demonstrate how to facilitate rule extraction from a deep neural network by retraining it in order to encourage sparseness in the weight matrices and make the hidden units be either maximally or minimally active. " \
       "Instead of using datasets which combine the attributes in an unclear manner, we show the effectiveness of the methods on the task of recon- structing predefined Boolean concepts so it can later be assessed to what degree the patterns were captured in the rule sets. " \
       "The evaluation shows that reducing the connectivity of the network in such a way significantly assists later rule extraction, and that when the neurons are either mini- mally or maximally active it suffices to consider one threshold per hidden unit."

text2 = "Classification rules and rules describing interesting subgroups are important components of descriptive machine learning. Rule learning algorithms typically proceed in two phases: " \
        "rule refinement selects con- ditions for specializing the rule, and rule selection selects the final rule among several rule candidates. While most conventional algorithms use the same heuristic for guiding both phases, " \
        "recent research in- dicates that the use of two separate heuristics is conceptually better justified, improves the coverage of positive examples, and may result in better classification accuracy. " \
        "The paper presents and evaluates two new beam search rule learning algorithms: DoubleBeam-SD for subgroup discovery and DoubleBeam-RL for classification rule learning. " \
        "The algorithms use two separate beams and can combine various heuris- tics for rule refinement and rule selection, which widens the search space and allows for finding rules with improved quality. " \
        "In the classification rule learning setting, the experimental results confirm previ- ously shown benefits of using two separate heuristics for rule refinement and rule selection. In subgroup discovery, " \
        "DoubleBeam-SD algorithm variants outperform several state-of-the-art related algorithms."
texts = [text1, text2]

# =============extract candidate chunks=================
def extract_candidate_chunks(text):

    import nltk, itertools, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # tokenize, POS-tag, and chunk using regular expressions
    grammar = "KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"
    chunker = nltk.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                      for sent in nltk.sent_tokenize(text))

    # ==== method 1======
    candidates_with_POS = []
    candidates = []
    tree_chunked_sents = [chunker.parse(tagged_sent) for tagged_sent in tagged_sents]
    for tree in tree_chunked_sents:
        for subtree in tree.subtrees():
            if subtree.label() == 'KT':
                candidates_with_POS.append(subtree.leaves())
    for cand in candidates_with_POS:
        NP = []
        for word, pos in cand:
            NP.append(word.lower())
        candidates.append(" ".join(NP))

    # ==== method 2======
    # BOI_tagged_chunked_sents = [nltk.tree2conlltags(tree_chunked_sent)
    #                             for tree_chunked_sent in tree_chunked_sents]
    # all_chunks = list(itertools.chain.from_iterable(BOI_tagged_chunked_sents))
    #
    # #   get all the NP Chunk and exclude all the non-NP chunk
    # groups = []
    # for key, group in itertools.groupby(all_chunks, lambda x : x[2]!='O'):
    #     if(key):
    #         groups.append(list(group))
    #
    # # get all the candidate except stopwords and punkt
    # candidates = [" ".join(word for word, pos, chunk in group).lower()
    #               for group in groups]
    candidates = [candidate for candidate in candidates
                  if candidate not in stop_words
                  and not all(char in punct for char in candidate)]
    return candidates


# =============extract candidate keywords, formulate [ WordNet ]=================
def extract_candidate_words(text):
    import itertools, nltk, string
    good_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS']

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # tokenize and POS-tag words
    sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    tagged_sentences = nltk.pos_tag_sents(sentences)
    tagged_words = itertools.chain.from_iterable(tagged_sentences)

    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if(tag in good_tags) and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


# =============score kandidate by tf*idf=================
def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    import gensim, nltk
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
    #print(list(corpus_tfidf))
    return corpus_tfidf, dictionary

    # ===== test  score_keyphrases_by_tfidf() =======
# corpus_tfidf, dictionary =  score_keyphrases_by_tfidf(texts)
# results = []
# for doc in corpus_tfidf:
#     for key, tfidf in doc:
#         results.append((tfidf, dictionary[key]))
# print(sorted(results))

# ================score key phrases by textrank=================
def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee
    import networkx, nltk

    # tokenize for all words, and extract *candidate* words
    words = [word.lower() for sent in nltk.sent_tokenize(text)
                            for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)

    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))

    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a,b = tee(iterable, 2)
        next(b)
        return zip(a,b)

    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords <1:
        n_keywords = int(round(len(candidates)))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda  x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words)/float(len(kp_words))
            keyphrases[" ".join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)

# r = score_keyphrases_by_textrank(text1)
# print(list(r))


# =============== supervised algorithm to extract keyphrases ============
def extract_candidate_features(candidates, doc_text, doc_excerpt, doc_title):
    import collections, math, nltk, re

    candidate_scores = collections.OrderedDict()

    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                            for word in nltk.word_tokenize(sent))

    for candidate in candidates:
        # ???????
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)

        # frequency-based
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))

        # count could be 0 for multiple reasons; shit happens in a simplified example
        if not cand_doc_count:
            print('**WARNING:', candidate, 'not found!')
            continue

        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)

        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(w for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0


        # positional
        # found in title, key excerpt
        in_title = 1 if pattern.search(doc_title) else 0
        in_excerpt = 1 if pattern.search(doc_excerpt) else 0

        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_mathch = pattern.search(doc_text)
        abs_first_occurrence = first_mathch.start() / doc_text_length
        if cand_doc_count == 1:
            spread = 0.0
            abs_last_occurrence = abs_first_occurrence
        else:
            for last_match in pattern.finditer(doc_text):
                pass
            abs_last_occurrence = last_match.start() / doc_text_length

            spread = abs_last_occurrence - abs_first_occurrence

        candidate_scores[candidate] = {'term_count' : cand_doc_count,
                                       'term_length' : term_length,
                                       'max_word_length': max_word_length,
                                       'spread': spread,
                                       'in_excerpt': in_excerpt, 'in_title': in_title,
                                       'abs_first_occurrence': abs_first_occurrence,
                                       'abs_last_occurrence': abs_last_occurrence}
        return  candidate_scores