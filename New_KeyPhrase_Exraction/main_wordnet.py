from New_KeyPhrase_Exraction.reader import *
from New_KeyPhrase_Exraction.tfidf_method import *
from New_KeyPhrase_Exraction.wordnet_method import *
import itertools



path = "/Users/apple/Documents/tu_darmstadt/Masterarbeit/Documents/Fuernkranz Johannes"
texts = reader(path)
texts_dic = pre_processing(texts)

# get all the titles, abstracts, keywords
titles, abstracts, keywords = get_titles_abstracts_keywords(texts_dic)
abstracts = " ".join(abstracts)
#print(abstracts)

keyphrase = score_keyphrases_by_textrank(abstracts)

#improved method, score again according to the keywords and abstract
titles_list, keywords_list = get_titles_keywords_by_words(titles, keywords)
keyphrase = score_again_to_candidate(keyphrase, keywords_list,titles_list )
keyphrase = sorted(keyphrase, key = lambda x: x[1], reverse=True)

print(keyphrase)

