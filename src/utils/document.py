import unicodedata
import nltk
import utils.text

class Document:
    def __init__(self, doc_name, doc_id):
        self.name = doc_name
        self.id = doc_id
        self.tokens = []
        self.mentions = []
        self.sentences = []

class MentionFromDict:
    '''
    This supports loading wikilinks directly from json. It provides backward compatibility for our code and allows
    to load preprocessed datasets from disk when there's no need to access the entire document
    '''
    def __init__(self, d, document):
        self._d = d
        self._document = document

    def document(self):
        return self._document

    def gold_sense_url(self):
        return self._d['wikiurl']

    def gold_sense_id(self):
        return self._d['wikiId']

    def mention_text(self):
        return self._d['word']

    def mention_text_tokenized(self):
        return self._d['mention_as_list']

    def left_context(self, max_len=None):
        return self._d['left_context'] if max_len is None or len(self._d['left_context']) < max_len \
            else self._d['left_context'][:max_len]

    def left_context_iter(self):
        for x in self._d['left_context'][::-1]:
            yield x

    def right_context(self, max_len=None):
        return self._d['right_context'] if max_len is None or len(self._d['right_context']) < max_len \
            else self._d['right_context'][:max_len]

    def right_context_iter(self):
        for x in self._d['right_context']:
            yield x


class Mention:
    def __init__(self, document, mention_start, mention_end, gold_sense_id=None, gold_sense_url=None):
        self._document = document
        self._mention_start = mention_start
        self._mention_end = mention_end
        self._gold_sense_id = gold_sense_id
        self._gold_sense_url = gold_sense_url

        self.candidates = None
        self.predicted_sense = None
        self.predicted_sense_global = None

    def document(self):
        return self._document

    def gold_sense_url(self):
        return self._gold_sense_url

    def gold_sense_id(self):
        return self._gold_sense_id

    def mention_text(self):
        return ' '.join(self.mention_text_tokenized())

    def mention_text_tokenized(self):
        return self.document().tokens[self._mention_start: self._mention_end]

    def left_context(self, max_len=None):
        l = [t for t in self.left_context_iter()]
        l = l if max_len is None or len(l) <= max_len else l[0:max_len]
        l.reverse()
        return l

    def left_context_iter(self):
        if self._mention_start > 0:
            for t in self.document().tokens[self._mention_start - 1:: -1]:
                yield t

    def right_context(self, max_len=None):
        l = [t for t in self.right_context_iter()]
        return l if max_len is None or len(l) <= max_len else l[0:max_len]

    def right_context_iter(self):
        if self._mention_end < len(self.document().tokens):
            for t in self.document().tokens[self._mention_end:]:
                yield t


class Token:
    def __init__(self, text):
        self.text = text
        self.pos = None