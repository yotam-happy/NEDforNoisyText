import mysql.connector
import re
import urllib
import utils.text
import pickle

class WikipediaDbWrapperCached:
    """
    Cached version of the Db object
    """

    def __init__(self, concept_filter=None):
        """
        All the parameters are self explanatory...
        """
        self.concept_filter = concept_filter

        self._cache_page_title = dict()
        self._cache_page_resolve = dict()

    def fillCache(self, otherdb, user, password, database, host='127.0.0.1'):
        self._cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self._cursor = self._cnx.cursor(buffered=True)
        query = "SELECT page_title, page_id FROM page"
        self._cursor.execute(query)

        row = self._cursor.fetchone()
        i = 0
        while row is not None:
            page_title = str(row[0])
            page_id = int(row[1])
            page_id_resolved = otherdb.resolvePage(str(page_title))

            title_for_lookup = urllib.unquote(page_title)
            title_for_lookup = utils.text.strip_wiki_title(title_for_lookup)

            self._cache_page_title[page_id] = page_title

            if page_id_resolved is not None:
                self._cache_page_resolve[title_for_lookup] = page_id_resolved

            i += 1
            if i % 1000 == 0:
                print "done", i

            row = self._cursor.fetchone()

    def getPageTitle(self, page_id):
        return self._cache_page_title[page_id] if page_id in self._cache_page_title else None

    def resolvePage(self, title, verbose=False, print_errors=True, use_pagelink_table=False):
        title_for_lookup = urllib.unquote(title)
        title_for_lookup = utils.text.strip_wiki_title(title_for_lookup)
        return self._cache_page_resolve[title_for_lookup] if title_for_lookup in self._cache_page_resolve else None

    def load(self, path):
        with open(path, 'r') as f:
            self._cache_page_resolve = pickle.load(f)
            self._cache_page_title = pickle.load(f)

    def save(self, path):
        with open(path, 'w') as f:
            pickle.dump(self._cache_page_resolve, f)
            pickle.dump(self._cache_page_title, f)
