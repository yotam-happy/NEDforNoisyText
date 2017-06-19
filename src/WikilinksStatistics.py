import json
import math
import random
import sys
import unicodedata

import nltk
import numpy as np
from nltk.corpus import stopwords
import utils.text
import math
import re

class WikilinksStatistics:
    """
    This class can calculate a number of statistics regarding the
    wikilink dataset.

    To calculate the statistics one needs to call calcStatistics() method.
    The class will then populate the following member variables:

    mentionCounts       dictionary of mention=count. Where mention is a surface term to be disambiguated
                        and count is how many times it was seen n the dataset
    conceptCounts       dictionary of concept=count. Where a concept is a wikipedia id (a sense), and count
                        is how many times it was seen (how many mentions refered to it)
    contextDictionary   dictionary of all words that appeared inside some context (and how many times)
    mentionLinks        holds for each mention a dictionary of conceptsIds it was reffering to and how many
                        times each. (So its a dictionary of dictionaries)
    """

    def __init__(self, wikilinks_iter, load_from_file_path=None, entity_transform=None):
        """
        Note the statistics are not calculated during init. (unless loaded from file)
        so must explicitly call calcStatistics()
        :param wikilinks_iter:      Iterator to a dataset
        :param load_from_file_path: If given then the statistics are loaded from this file
        """
        self._wikilinks_iter = wikilinks_iter
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.conceptCounts2 = dict()
        self.contextDictionary = dict()
        self.entity_transform = entity_transform
        if load_from_file_path is not None:
            self.loadFromFile(load_from_file_path)
            if self.entity_transform is not None:
                self.apply_transform()

    def apply_transform(self):
        """
        Applies entity transform to get mentionLinks and conceptCounts after transform
        """
        transformedConceptCounts = dict()
        for concept, count in self.conceptCounts.iteritems():
            transformed = self.entity_transform.entity_id_transform(int(concept))
            if transformed is not None:
                transformedConceptCounts[transformed] = transformedConceptCounts.get(transformed, 0) + int(count)
        self.conceptCounts = transformedConceptCounts

        transformedLinks = dict()
        for mention, links in self.mentionLinks.iteritems():
            transformedLinks[mention] = dict()
            for concept, count in links.iteritems():
                transformed = self.entity_transform.entity_id_transform(int(concept))
                if transformed is not None:
                    transformedLinks[mention][transformed] = transformedLinks[mention].get(transformed, 0) + int(count)
        self.mentionLinks = transformedLinks

    def getCandidateConditionalPrior(self, concept, mention):
        mention_text = utils.text.normalize_unicode(mention.mention_text())
        if mention_text not in self.mentionLinks or concept not in self.mentionLinks[mention_text]:
            return 0
        return float(self.mentionLinks[mention_text][concept]) / np.sum(self.mentionLinks[mention_text].values())

    def getCandidatePrior(self, concept):
        return float(self.conceptCounts[concept]) if concept in self.conceptCounts else 0

    def getMostProbableSense(self, mention):
        if len(mention.candidates) == 0:
            return None
        counts = {cand: self.getCandidateConditionalPrior(cand, mention) for cand in mention.candidates}
        return max(counts.iterkeys(), key=(lambda key: counts[key]))

    def getSensesFor(self, l):
        return {s for w in l for s in self.getCandidatesForMention(w)}

    def saveToFile(self, path):
        """ saves statistics to a file """
        f = open(path, mode='w')
        f.write(json.dumps(self.mentionCounts)+'\n')
        f.write(json.dumps(self.mentionLinks)+'\n')
        f.write(json.dumps(self.conceptCounts)+'\n')
        f.write(json.dumps(self.conceptCounts2)+'\n')
        f.write(json.dumps(self.contextDictionary)+'\n')
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = dict() #json.loads(l[0])
        self.conceptCounts = {int(a): int(b) for a, b in json.loads(l[2]).iteritems()}

        self.mentionLinks = dict()
        for m, ll in json.loads(l[1]).iteritems():
            self.mentionLinks[m] = dict()
            for c, cc in ll.iteritems():
                self.mentionLinks[m][int(c)] = int(cc)

        self.conceptCounts2 = dict() #json.loads(l[3])
        self.contextDictionary = json.loads(l[4])
        f.close()

    def calcStatisticsFromDBPedia(self, page_link_path, lexicalization_path, db):
        k = 0
        with open(page_link_path) as f:
            for i, line in enumerate(f):
                if i > 0 and i % 1000 == 0:
                    print "done", i, "resolved", float(k) / i
                m = re.findall('(<[^>]*>)', line)
                if m is None or len(m) != 3:
                    continue
                ent = m[2][m[2].rfind('/')+1:-1]
                if ent.find(':') > -1: # used for categories, dont try and resolve this
                    continue
                id = db.resolvePage(ent)
                if id is None:
                    continue
                k += 1
                self.conceptCounts[id] = self.conceptCounts.get(id, 0) + 1

        k = 0
        with open(lexicalization_path) as f:
            for i, line in enumerate(f):
                if i > 0 and i % 10000 == 0:
                    print "done", i, "resolved", float(k) / i
                m = line.split('\t')
                if len(m) != 3:
                    continue
                ent = m[1][m[1].rfind('/')+1:]
                ent_id = db.resolvePage(ent)
                if ent_id is None:
                    continue

                mention = utils.text.normalize_unicode(m[0])
                if mention not in self.mentionLinks:
                    self.mentionLinks[mention] = dict()
                self.mentionLinks[mention][ent_id] = self.mentionLinks[mention].get(ent_id, 0) + int(m[2])
                k += 1


    def calcStatistics(self):
        """
        calculates statistics and populates the class members. This should be called explicitly
        as it might take some time to complete. It is better to call this method once and save
        the results to a file if the dataset is not expected to change
        """
        print "getting statistics"
        for wlink in self._wikilinks_iter.jsons():
            mention_text = utils.text.normalize_unicode(wlink['word'])

            if mention_text not in self.mentionLinks:
                self.mentionLinks[mention_text] = dict()
            self.mentionLinks[mention_text][wlink['wikiId']] = self.mentionLinks[mention_text].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[mention_text] = self.mentionCounts.get(mention_text, 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in wlink['right_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in wlink['left_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

        # counts mentions per concept
        for mention, entities in self.mentionLinks.iteritems():
            for entity in entities.keys():
                self.conceptCounts2[entity] = self.conceptCounts2.get(entity, 0) + 1

    def getCandidatesForMention(self, mention, p=0, t=0):
        """
        Returns the most probable sense + all other candidates where p(candidate|mention)>=p
        and with at least t appearances

        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        mention_text = utils.text.normalize_unicode(
            mention.mention_text() if hasattr(mention, 'mention_text') else mention)
        if mention_text not in self.mentionLinks:
            return set()
        candidates = self.mentionLinks[mention_text]
        tot = sum([y for x, y in candidates.iteritems()])
        out = dict()

        for x, y in candidates.iteritems():
            if float(y) / tot >= p and y > t:
                out[x] = y

        return {int(x) for x, y in out.iteritems()}

    def getGoodMentionsToDisambiguate(self, p=0, t=0):
        """
        Returns a set of mentions that are deemed "good"
        :param f:
        :return:
        """

        # take those mentions where the second +
        # most common term appears more then f times
        s = set()
        for mention, candidates in self.mentionLinks.iteritems():
            if len(self.getCandidatesForMention(mention, p=p, t=t)) >= 2:
                s.add(mention)
#            tot = sum([y for x, y in candidates.iteritems()])
#            max_y = 0
#            for x, y in candidates.iteritems():
#                if y > max_y:
#                    max_y = y
#            if max_y >= t and float(max_y) / tot <= p:
#                s.add(mention)
        return s

    def prettyPrintMentionStats(self, m, db):
        try:
            s = "["
            for x, y in m.iteritems():
                t = db.getPageInfoById(x)[2]
                s += str(t) + ": " + str(y) + "; "
            s += ']'
            print s
        except :
            print "Unexpected error:", sys.exc_info()[0]
            print m

    def _sortedList(self, l):
        l = [(k,v) for k,v in l.items()]
        l.sort(key=lambda (k, v): -v)
        return l

    def printSomeStats(self):
        """
        Pretty printing of some of the statistics in this object
        """

        print "distinct terms: ", len(self.mentionCounts)
        print "distinct concepts: ", len(self.conceptCounts)
        print "distinct context words: ", len(self.contextDictionary)

        k, v = self.mentionLinks.items()[0]
        wordsSorted = [(k, self._sortedList(v), sum(v.values())) for k,v in self.mentionLinks.items()]
        wordsSorted.sort(key=lambda (k, v, d): v[1][1] if len(v) > 1 else 0)
        print("some ambiguous terms:")
        for w in wordsSorted[-10:]:
            print w

if __name__ == "__main__":
    from WikilinksIterator import WikilinksNewIterator
    import os
    from PPRforNED import *
    from DbWrapper import WikipediaDbWrapper

    db_user = sys.argv[1]
    db_pass = sys.argv[2]
    db_schema = sys.argv[3]
    db_host = sys.argv[4]
    wikipedia_corpus_dir = 'data/intralinks/all'

    print 'calculate wikipedia corpus statistics'
    it = WikilinksNewIterator(wikipedia_corpus_dir)
    stats = WikilinksStatistics(it)
    stats.calcStatistics()
    stats.saveToFile(os.path.join(wikipedia_corpus_dir, 'all_stats'))
    stats.printSomeStats()

    print 'calculate PPRforNED stats'
    ppr_itr = PPRIterator(path='data/PPRforNED/AIDA_candidates')
    ppr_stats = PPRStatistics(ppr_itr)
    wikiDB = WikipediaDbWrapper(user=db_user, password=db_pass, database=db_schema, host=db_host)
    ppr_stats.calcStatistics(wikiDB)
    print len(ppr_stats.conceptNames)
    ppr_stats.saveToFile('/data/PPRforNED/ppr_stats')