import threading
from queue import Queue
from Crawl_Webpage.spider import Spider
from Crawl_Webpage.general import *
from Crawl_Webpage.domain import *

PROJECT_NAME = 'parser_link'

# TODO change
# HOMEPAGE = 'http://dblp.uni-trier.de/pers/hd/f/F=uuml=rnkranz:Johannes'
HOMEPAGE = 'http://www.sciencedirect.com/science/article/pii/S0957417417301914?via%3Dihub'
DOMAIN_NAME = get_domain_name(HOMEPAGE)
QUEUE_FILE = PROJECT_NAME + '/queue.txt'
CRAWLED_FILE = PROJECT_NAME + '/crawled.txt'
NUMBER_OF_THREADS = 8

# for thread
queue = Queue()
Spider(PROJECT_NAME, HOMEPAGE, DOMAIN_NAME)



# Check if there are items in the queue, if so crawl them
def crawl():
    queued_links = file_to_set(QUEUE_FILE)
    if len(queued_links) > 0:
        print(str(len(queued_links + ' links in the queue')))
        create_jobs()


# Each queued link is a new job
def create_jobs():
    for link in file_to_set(QUEUE_FILE):
        queue.put(link)
    queue.join()
    crawl()

Spider.crawl_page(threading.current_thread().name, HOMEPAGE)
