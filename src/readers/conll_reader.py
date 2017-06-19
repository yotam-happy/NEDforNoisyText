from utils.document import *


def _CoNLLFileToDocIterator(fname, split='testb'):
    f = open(fname,'r')
    lines = f.readlines()

    curdocName = None
    curdocSplit = None
    curdoc = None

    for line in lines:
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            if curdocName is not None and \
                    (split == 'all'
                     or curdocSplit == split
                     or split.startswith('not_') and not split.endswith(curdocSplit)):
                yield (curdoc, curdocName)
            sp = line.split(' ')
            curdocName = sp[2][:-1]
            curdocSplit = 'testa' if sp[1].endswith('testa') else ('testb' if sp[1].endswith('testb') else 'train')
            curdoc = []
        else:
            curdoc.append(line)
    if curdocName is not None and (split is None or curdocSplit == split):
        yield (curdoc, curdocName)


def _CoNLLRawToTuplesIterator(lines):
    '''
    yields tuples:
    (surface,ismention,islinked,YAGO2,WikiURL,WikiId,FB)
    surface is either a word or the full mention
    '''
    for line in lines:
        if len(line) == 0:
            # sentence boundary.
            continue
        t = line.split('\t')
        if len(t) == 1:
            yield (t[0], t[0], False, None, None, None, None, None)
        else:
            if t[1] != 'B':
                continue
            if t[3] == '--NME--':
                yield (t[2], True, False, None, None, None, None)
            else:
                yield (t[2], True, True, t[3], t[4], int(t[5]), t[6] if len(t) >= 7 else None)

class CoNLLIterator:
    # the new iterator does not support using a zip file.
    def __init__(self, fname, db, split='testa', include_unresolved=False):
        self._fname = fname
        self._split = split
        self._db = db
        self._include_unresolved = include_unresolved

    def documents(self):
        i = 0
        for (doc_lines, doc_name) in _CoNLLFileToDocIterator(self._fname, self._split):
            doc = Document(doc_name, i)
            i += 1
            mention = None
            sent = []
            for line in doc_lines:
                if len(line) == 0:
                    # sentence boundary.
                    doc.sentences.append(' '.join(sent))
                    sent = []
                    continue
                t = line.split('\t')

                sent.append(t[0])
                doc.tokens.append(t[0])
                if len(t) == 1:
                    continue

                if t[1] != 'I' and mention is not None:
                    mention = None
                if t[1] == 'I' and (t[3] != '--NME--' or self._include_unresolved):
                    mention._mention_end += 1

                if t[1] == 'B' and (t[3] != '--NME--' or self._include_unresolved):
                    if t[3] != '--NME--':
                        gold_sense_url = t[4]
                        gold_sense_id = self._db.resolvePage(gold_sense_url[gold_sense_url.rfind('/')+1:])
                        mention = Mention(doc, len(doc.tokens) - 1, len(doc.tokens),
                                          gold_sense_id=gold_sense_id, gold_sense_url=gold_sense_url)
                    else:
                        mention = Mention(doc, len(doc.tokens) - 1, len(doc.tokens))
                    doc.mentions.append(mention)

            yield doc

    def mentions(self):
        for doc in self.documents():
            for mention in doc.mentions:
                yield mention
