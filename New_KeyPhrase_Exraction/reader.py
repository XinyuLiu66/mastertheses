import os
from nltk.stem import WordNetLemmatizer, PorterStemmer
port = PorterStemmer()

# read all the documents information from a specified directory(Folder)
def reader(path):
    documents = []
    files = os.listdir(path)
    files = [f for f in files if f.endswith("txt")]
    for file in files:
        doc = []
        if os.path.isdir(file):
            continue
        else:
            f = open(path + "/" + file,'rt')

            for line in f:
                doc.append(line.replace("\n",""))
        documents.append(doc)
    return documents

# data pre-processing, using LEMMATIZATION to reduce inflectional forms and
# sometimes derivationally related forms of a word to a common base form.
# data pre-processing, store the data from reader in a dictionary, e.g {title:" ", abstract:" ",keywords:" "}
def pre_processing(documents):
    results = list()
    for doc in documents:
        res = {}
        res["title"] = ""
        res["abstract"] = ""
        res["keywords"] = list()
        for ele in doc:
            # change e.g. <Title:Class Binarization> to Title:Class Binarization
            str = ele.replace("<","")
            str = str.replace(">", "")

            # get content under each tag
            if("Title:" in str):
                str = str.replace("Title:","").lower()
                # lemma
                str = " ".join(WordNetLemmatizer().lemmatize(w) for w in str.split(" "))
                res["title"] = str
            elif("Abstract" in str):
                str = str.replace("Abstract:", "").lower()
                str = " ".join(WordNetLemmatizer().lemmatize(w) for w in str.split(" "))
                res["abstract"] = str
            if ("Keywords:" in str):
                keywords = (str.replace("Keywords:", "").lower()).split(";")
                keywords = [WordNetLemmatizer().lemmatize(w) for w in keywords]
                res["keywords"] = keywords
        results.append(res)
    return results







# demo

# path = "/Users/apple/Documents/tu_darmstadt/Masterarbeit/Documents/Fuernkranz Johannes"
# texts = reader(path)
# texts_dic = pre_processing(texts)
# print(texts_dic)


