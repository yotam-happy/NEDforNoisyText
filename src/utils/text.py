import unicodedata
import re
from unidecode import unidecode

def normalize_unicode(s, lower=True):
    if type(s) == bytearray:
        return unidecode(unicode(str(s), 'utf-8')).lower() if lower else unidecode(unicode(str(s), 'utf-8'))
    elif type(s) == unicode:
        return unidecode(s).lower() if lower else unidecode(s)
    else:
        return unidecode(unicode(s, 'utf-8')).lower() if lower else unidecode(unicode(s, 'utf-8'))

def strip_wiki_title(title):
    t = re.sub('[^0-9a-zA-Z]+', '_', normalize_unicode(title))
    if len(t) > 1 and t[0] == '_':
        t = t[1:]
    if len(t) > 1 and t[-1] == '_':
        t = t[:-1]
    return t
