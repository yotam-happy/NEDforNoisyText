# contains code to handle wikipedia xml dump

import urllib
import re
import os
import fnmatch
import nltk
import sys
from nltk.corpus import stopwords
import random
import json
from utils.text import normalize_unicode

def wikipedia_dump_iterator(dump_dir, printerr=None):
    doc_id = None
    doc_title = None
    lines = []
    for root, dirnames, filenames in os.walk(dump_dir, onerror=printerr):
        for filename in fnmatch.filter(filenames, 'wiki_*'):
            with open(os.path.join(root, filename),'r') as f:
                for line in f:
                    if line.startswith('<doc id='):
                        doc_id = line[line.find('id="') + 4: line.find('"', line.find('id="') + 4)]
                        doc_title = line[line.find('title="') + 7: line.find('"', line.find('title="') + 7)]
                        lines = []
                    elif line.startswith('</doc>'):
                        yield doc_id, doc_title, lines
                    else:
                        lines.append(line)

def wikipedia_doc_tokenizer(doc_lines):
    doc = '\n'.join(doc_lines)
    doc = re.sub('</?a.*?>', '', doc)
    doc = doc.decode('utf-8').encode('ascii', 'ignore').lower()
    tokens = nltk.word_tokenize(doc)
    for token in tokens:
        yield token

class wlink_writer:
    def __init__(self, dir, json_per_file=400000):
        self._dir = dir
        self._n = 0
        self._json_per_file = json_per_file
        if not os.path.exists(dir):
            os.mkdir(dir)
        self._l = []

    def _next_file(self):
        f = open(os.path.join(self._dir, 'wikilinks_{}.json'.format(self._n)), mode='w')
        self._n += 1
        print 'creating file n', self._n
        return f

    def _dump(self):
        if len(self._l) >= 1:
            f = self._next_file()
            for s in self._l:
                f.write(s + '\n')
            f.close()
            self._l = []

    def save(self, wlink):
        self._l.append(json.dumps(wlink))
        if len(self._l) >= self._json_per_file:
            self._dump()

    def finalize(self):
            self._dump()

def wikipedia_crossref_iterator(dump_dir):
    for doc_id, doc_title, doc_lines in wikipedia_dump_iterator(dump_dir):
        for line in doc_lines:
            line = normalize_unicode(line, lower=False)
            for m in re.finditer('<a href[^<]*</a>', line):
                tag = line[m.start(): m.end()]
                target = tag[9: tag[9:].find('"') + 9]
                mention = tag[tag[9:].find('">') + 11: -4]
                yield mention, target, line[:m.start()], line[m.end() + 1:]

if __name__ == "__main__":
    if sys.argv[1] == 'w2v_train':
        print 'creating w2v training...'
        input_dir = sys.argv[2]
        output_file = sys.argv[3]
        tokens = []
        stoplist = set(stopwords.words('english'))
        with open(output_file, 'w+') as f:
            for i, (doc_id, doc_title, doc_lines) in enumerate(wikipedia_dump_iterator(input_dir)):
                if i % 100000 == 0:
                    print 'done', i, 'documents'
                for token in wikipedia_doc_tokenizer(doc_lines):
                    if (len(token) < 2 and token not in stoplist) or not re.search('[a-zA-Z]', token):
                        continue
                    tokens.append(token + ' ' + str(doc_id) + '\n')
                    if len(tokens) > 100 * 1000 * 1000:
                        # partially shuffle the dataset - shuffle every 100M tokens (much smaller then entire dataset...)
                        #random.shuffle(tokens)
                        f.writelines(tokens)
                        tokens = []
            if len(tokens) > 0:
                random.shuffle(tokens)
                f.writelines(tokens)
                tokens = []
    elif sys.argv[1] == 'intralink_jsons':
        from WikilinksIterator import WikilinksNewIterator
        from DbWrapper import WikipediaDbWrapper

        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        db_user = sys.argv[4]
        db_pass = sys.argv[5]
        db_schema = sys.argv[6]
        db_host = sys.argv[7]
        wikiDB = WikipediaDbWrapper(user=db_user, password=db_pass, database=db_schema, host=db_host)

        writer = wlink_writer(output_dir + '_tmp')
        for mention, target, left_context, right_context in wikipedia_crossref_iterator(input_dir):
            wlink = dict()
            wlink['wikiurl'] = target
            wlink['word'] = mention
            wlink['left_context'] = left_context
            wlink['right_context'] = right_context
            writer.save(wlink)
        writer.finalize()

        it = WikilinksNewIterator(output_dir + '_tmp', resolveIds=True, db=wikiDB)
        writer = wlink_writer(output_dir)
        for wlink in it.jsons():
            writer.save(wlink)
        writer.finalize()

    else:
        'unknown task'