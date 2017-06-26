import os
from zipfile import ZipFile
import pandas as pd # pandas
import re
import pickle
import ujson as json
import nltk
import unicodedata
from nltk.corpus import stopwords
from utils.document import *
import time
import urllib

class WikilinksOldIterator:
    """
    This iterator is meant to be used with the older format of the dataset where each
    file has a single json with many wikilinks in it. It is arguably faster then the
    new iterator but requires quite a lot of memory

    note that WikilinksNewIterator and WikilinksOldIterator can be dropped-in-replacements of each other
    """

    # path should either point to a zip file or a directory containing all dataset files,
    def __init__(self, path="wikilink.zip", limit_files = 0):
        """
        :param path:    Path to either a zip file or a directory containing the dataset
        """
        self._path = path
        self.limit_files =  limit_files
        self.wikilink_fname = 'wikilink.zip'

    def get_wlinks(self):
        # outputs the next wlink piece
        # print "get next()"
        for wlink in self._wikilinks_iter.wikilinks():
            yield wlink

    def _wikilink_files(self):
        if os.path.isdir(self._path):
            for file in os.listdir(self._path):
                # if os.path.isdir(os.path.join(self._path, file)):
                #     continue
                # print "opening ", file
                # yield open(os.path.join(self._path, file), 'r')
                if file != self.wikilink_fname: # assuming zip file
                    continue
                zf = ZipFile(self._path+'\\'+self.wikilink_fname, 'r') # Read in a list of zipped files
                for fname in zf.namelist():
                    print "opening ", fname
                    yield zf.open(fname)

        else: # assume zip
            zf = ZipFile(self._path, 'r') # Read in a list of zipped files
            for fname in zf.namelist():
                print "opening ", fname
                yield zf.open(fname)

    def addMetionToContext(self, wlink):
        mention = wlink['word']
    def wikilinks(self):
        """
        This is the main function - it is a generator that can be used as an iterator
        returning a single wikilink object at a time
        """
        c = 0
        for f in self._wikilink_files():
            df = pd.read_json(f)
            for wlink in df.wlinks:
                if(not 'wikiId' in wlink):
                    continue

                yield wlink
            df = None
            c += 1
            f.close()
            if self.limit_files > 0 and c == self.limit_files:
                print "stoppped at file ", self.limit_files
                break

class WikilinksNewIterator:

    # the new iterator does not support using a zip file.
    def __init__(self, path, limit_files = 0, mention_filter=None, resolveIds=False, db=None):
        self._path = path
        self._limit_files = limit_files
        self._mention_filter = mention_filter
        self._stopwords = stopwords.words('english')
        self._resolveIds = resolveIds
        self._db = db

    def get_wlink(self):
        # outputs the next wlink piece
        # print "get next()"
        for wlink in self.wikilinks():
            yield wlink

    def _wikilink_files(self):
        for i, f in enumerate(os.listdir(self._path)):
            if os.path.isdir(os.path.join(self._path, f)):
                continue
            print time.strftime("%H:%M:%S"), "- opening", f, "(", i, "opened so far in this epoch)"
            yield open(os.path.join(self._path, f), 'r')

    def jsons(self):
        r = 0
        t = 0
        for c, f in enumerate(self._wikilink_files()):
            lines = f.readlines()
            for line in lines:
                if len(line) > 0:
                    wlink = json.loads(line)

                    # filter
                    if not 'word' in wlink:
                        continue
                    if 'right_context' not in wlink and 'left_context' not in wlink:
                        continue
                    if self._mention_filter is not None and wlink['word'] not in self._mention_filter:
                        continue

                    if not self._resolveIds:
                        wlink['wikiId'] = int(wlink['wikiId']) if 'wikiId' in wlink else None
                    else:
                        t += 1
                        url = wlink['wikiurl']
                        if url.rfind('/') > -1:
                            url = url[url.rfind('/')+1:]
                        if url.find('#') > -1:
                            url = url[:url.find('#')]
                        url = urllib.unquote(url)
                        url = url.replace(' ', '_')

                        url = unicodedata.normalize('NFKD', url).encode('ascii', 'ignore')
                        if len(url) > 0 and url[0].islower():
                            tttt = list(url)
                            tttt[0] = tttt[0].upper()
                            url = ''.join(tttt)
                        wikiId = self._db.resolvePage(url)
                        if t % 50000 == 0:
                            print "% able to resolve:", 100 * float(r) / t, "% out of", t
                        if wikiId is None:
                            continue
                        else:
                            wlink['wikiId'] = wikiId
                            r += 1

                    if 'mention_as_list' not in wlink:
                        mention_as_list = unicodedata.normalize('NFKD', wlink['word']).encode('ascii','ignore').lower()
                        mention_as_list = nltk.word_tokenize(mention_as_list)
                        wlink['mention_as_list'] = mention_as_list

                    # preprocess context (if not already processed
                    if 'right_context' in wlink and not isinstance(wlink['right_context'], list):
                        wlink['right_context_text'] = wlink['right_context']
                        r_context = unicodedata.normalize('NFKD', wlink['right_context']).encode('ascii','ignore').lower()
                        wlink['right_context'] = nltk.word_tokenize(r_context)
                        wlink['right_context'] = [w for w in wlink['right_context']]
                    if 'left_context' in wlink and not isinstance(wlink['left_context'], list):
                        wlink['left_context_text'] = wlink['left_context']
                        l_context = unicodedata.normalize('NFKD', wlink['left_context']).encode('ascii','ignore').lower()
                        wlink['left_context'] = nltk.word_tokenize(l_context)
                        wlink['left_context'] = [w for w in wlink['left_context']]

                    # return
                    yield wlink

            f.close()
            if self._limit_files > 0 and c >= self._limit_files:
                break

    def mentions(self):
        for doc in self.documents():
            for mention in doc.mentions:
                yield mention

    def documents(self):
        for i, json in enumerate(self.jsons()):
            doc = Document(str(i), i)
            m = MentionFromDict(json, doc)
            doc.mentions.append(m)
            dd = [x for x in m.left_context_iter()]
            doc.tokens = dd[::-1] + \
                         [x for x in m.mention_text_tokenized()] + \
                         [x for x in m.right_context_iter()]
            doc.sentences = [' '.join(doc.tokens)]
            yield doc

if __name__ == "__main__":
    _path = "/home/yotam/pythonWorkspace/deepProject"
    from WikilinksIterator import *
    it = WikilinksNewIterator(_path+"/data/wikilinks/filtered/train")
    for i, wlink in enumerate(it.jsons()):
        txt = wlink['left_context_text'] + ' <<' + wlink['word'] + '>> ' + wlink['right_context_text']
        print txt
        if i > 100:
            break

