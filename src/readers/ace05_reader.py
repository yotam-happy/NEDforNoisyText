import os
from utils.document import *
import re


class Ace05Iterator:
    def __init__(self, folder, db, split='train'):
        self.db = db
        self.folder = folder
        self.split = split

    def _iter_files(self):
        for sp in ([self.split] if self.split != 'all' else ['train', 'test']):
            file_dir = os.path.join(self.folder, sp)
            gold_senses = os.path.join(self.folder, 'ace05-all-conll-wiki')
            for filename in os.listdir(file_dir):
                yield (os.path.join(file_dir, filename), os.path.join(gold_senses, filename + '-wiki'), filename)

    def _read_tags(self, fname, db):
        tags = dict()
        with open(fname, 'r') as f:
            raw = f.readlines()
        for line in raw:
            sp = line.strip().split('\t')
            if sp[0] == 'MISSING SENSE':
                continue
            wiki_title = sp[0][sp[0].rfind('/')+1:]
            doc_name = sp[1][:sp[1].rfind('.')]
            entity_marker = sp[2][1:]
            entity_id = db.resolvePage(wiki_title)
            if entity_id is not None:
                tags[(doc_name, entity_marker)] = entity_id
        return tags

    def mentions(self):
        for doc in self.documents():
            for mention in doc.mentions:
                yield mention

    def documents(self):
        pattern = re.compile('\([^()]*|\)')

        for k, (doc_fname, gold_fname, doc_name) in enumerate(self._iter_files()):
            with open(doc_fname, 'r') as f:
                doc_raw = f.readlines()
            with open(gold_fname, 'r') as f:
                gold_raw = f.readlines()

            doc = Document(doc_name, k)

            words = []
            for line in doc_raw[1:]:
                sp = line.strip().split('\t')
                if len(sp) <= 1:
                    continue
                words.append(sp[3].lower())

            mention_queue = []
            i = 0
            for line in gold_raw[1:]:
                if len(line.strip()) == 0:
                    continue
                match = re.findall(pattern, line.strip())
                for x in match:
                    if x.startswith('('):
                        x = x.strip('(*').replace('-LRB-', '(').replace('-RRB-', ')')
                        if x.rfind('#'):
                            x = x[x.rfind('#') + 1:]
                        mention_queue.append((i, x))
                    if x == ')':
                        mention_start, mention_entity = mention_queue.pop()
                        if mention_entity != '-EXCLUDE-' and mention_entity != '-NIL-':
                            ents = mention_entity.split('|')
                            ent_ids = {self.db.resolvePage(e) for e in ents if self.db.resolvePage(e) is not None}
                            if len(ent_ids) > 0:
                                doc.mentions.append(Mention(doc, mention_start, i+1, ent_ids, mention_entity))
                            else:
                                print 'cant find! :',  mention_entity
                i += 1

            doc.tokens = words
            yield doc
