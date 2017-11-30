from New_KeyPhrase_Exraction.reader import *
from New_KeyPhrase_Exraction.tfidf_method import *
import itertools

path = "/Users/apple/Documents/tu_darmstadt/Masterarbeit/Documents/Fuernkranz Johannes"
texts = reader(path)
texts_dic = pre_processing(texts)

# get all the titles, abstracts, keywords
titles, abstracts, keywords = get_titles_abstracts_keywords(texts_dic)
# implement using tf*idf method
print("Number of text is : ", len(abstracts), '\n')

corpus_tfidf, dictionary = score_keyphrases_by_tfidf(abstracts)

# sorted all the candidate chunks
all_chunks = list(itertools.chain.from_iterable(corpus_tfidf))


titles_list, keywords_list = get_titles_keywords_by_words(titles, keywords)

all_chunks = scroe_again_to_candidate(all_chunks, dictionary, keywords_list, titles_list)
all_chunks = sorted(all_chunks, key = lambda x: x[1], reverse=True)

# extract top 50 key phrase
NUM_OF_PHRASES = 50
keyPhrases = list()
for i in range(NUM_OF_PHRASES):
    keyPhrases.append( (dictionary[all_chunks[i][0]],  all_chunks[i][1]))
print(keyPhrases,"\n")


print(list(all_chunks))


