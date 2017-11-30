from  html.parser import HTMLParser
from urllib import parse

class Title_abs_keywords_finder(HTMLParser):

    atitle = False

    def __init__(self, base_url, page_url):
        super().__init__()
        self.base_url = base_url
        self.page_url = page_url
        self.title = ""
        self.abstract = ""
        self.keyword = ""



# When we call HTMLParser feed() this function is called when it encounters an opening tag <a>
    def handle_starttag(self, tag, attrs):
        if tag == 'span':
            for (attribute, value) in attrs:
                if attribute == 'class' and value == 'title-text':
                    Title_abs_keywords_finder.atitle = True

        else:
            pass

    # handle data
    def handle_data(self, data):
        if Title_abs_keywords_finder.atitle == True:
            print("Encountered some data  :", data)

    def handle_endtag(self, tag):
        if tag == 'span':
            Title_abs_keywords_finder.atitle  = False


# return all the page_links
    def getTitle(self):
        return self.title



    def error(self, message):
        pass

