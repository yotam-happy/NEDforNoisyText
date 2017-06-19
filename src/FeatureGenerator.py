import json

import numpy as np
import pandas as pd

import utils.text
from nltk.metrics.distance import edit_distance

class FeatureGenerator:
    def __init__(self, feature_names={}, stats=None, db=None, w2v=None,
                 yamada_embedding_path=None, yamada_id2title_path=None,
                 models_as_features=None):
        self.n = 0
        self.nn = 0
        self._stats = stats
        self._db = db
        self._w2v = w2v
        self.feature_names = [x for x in feature_names]
        print self.feature_names
        self.models_as_features = models_as_features
        self.models_as_features_predictors = {name: m.getPredictor() for name, m in self.models_as_features.iteritems()}

    def getPointwiseFeatureList(self):
        return self.feature_names

    def numPointwiseFeatures(self):
        return len(self.feature_names)

    def getFeatureVector(self, mention, entity):
        features = []

        page_title = self._db.getPageTitle(entity)
        page_title = utils.text.normalize_unicode(page_title) if page_title is not None else None
        mention_text = utils.text.normalize_unicode(mention.mention_text())

        for feature in self.feature_names:

            # Count features
            if feature == 'prior':
                features.append(self._stats.getCandidatePrior(entity))
            elif feature == 'prior_yamada':
                features.append(self._stats.getCandidatePriorYamadaStyle(entity))
            elif feature == 'normalized_prior':
                features.append(self._stats.getCandidatePrior(entity, normalized=True))
            elif feature == 'normalized_log_prior':
                features.append(self._stats.getCandidatePrior(entity, normalized=True, log=True))
            elif feature == 'relative_prior':
                if entity in mention.candidates:
                    count = 0
                    for cand in mention.candidates:
                        count += self._stats.getCandidatePrior(cand)
                    if count == 0:
                        features.append(float(0))
                    else:
                        features.append(float(self._stats.getCandidatePrior(entity)) / count)
                else:
                    features.append(float(0))
            elif feature == 'cond_prior':
                features.append(self._stats.getCandidateConditionalPrior(entity, mention))
            elif feature == 'n_of_candidates':
                features.append(len(mention.candidates))
            elif feature == 'max_prior':
                max_prior = self._stats.getCandidateConditionalPrior(entity, mention)
                for m in mention.document().mentions:
                    if entity in m.candidates and self._stats.getCandidateConditionalPrior(entity, m) > max_prior:
                        max_prior = self._stats.getCandidateConditionalPrior(entity, m)
                features.append(max_prior)

            # string similarity features
            elif feature == 'entity_title_starts_or_ends_with_mention':
                x = 1 if page_title is not None and (page_title.lower().startswith(mention_text.lower()) or page_title.lower().endswith(mention_text.lower())) else 0
                features.append(x)
            elif feature == 'mention_text_starts_or_ends_with_entity':
                x = 1 if page_title is not None and (mention_text.lower().startswith(page_title.lower()) or mention_text.lower().endswith(page_title.lower())) else 0
                features.append(x)
            elif feature == 'edit_distance':
                features.append(edit_distance(page_title.lower(), mention_text.lower()) if page_title is not None else 0)

            # context similarity features
            elif feature == 'yamada_context_similarity':
                if not hasattr(mention.document(), 'yamada_context_nouns'):
                    mention.document().yamada_context_nouns = \
                        self._opennlp.list_nouns(mention.document().sentences)

                if not hasattr(mention.document(), 'yamada_context_embd'):
                    mention.document().yamada_context_embd = dict()
                if mention_text not in mention.document().yamada_context_embd:
                    context_embd = self.yamada_txt_to_embd.text_to_embedding(
                        mention.document().yamada_context_nouns, mention_text)
                    mention.document().yamada_context_embd[mention_text] = context_embd
                context_embd = mention.document().yamada_context_embd[mention_text]
                entity_embd = self.yamada_txt_to_embd.from_the_cache(entity)

                self.n += 1
                if entity_embd is not None:
                    s = self.yamada_txt_to_embd.similarity(context_embd, entity_embd)
#                    print self.yamada_txt_to_embd.similarity(context_embd, entity_embd)
                    features.append(s)
                    if s > 0:
                        self.nn += 1
                else:
                    #print 0
                    features.append(0.0)

                if self.n % 100 == 0:
                    print "yamada got sim", self.nn / float(self.n)

            elif feature == 'our_context_similarity':
                if not hasattr(mention.document(), 'our_context_nouns'):
                    mention.document().our_context_nouns = \
                        self._w2v.get_nouns(mention.document().sentences)

                if not hasattr(mention.document(), 'our_context_embd'):
                    mention.document().our_context_embd = dict()
                if mention_text not in mention.document().our_context_embd:
                    context_embd = self._w2v.text_to_embedding(
                        mention.document().our_context_nouns, mention_text)
                    mention.document().our_context_embd[mention_text] = context_embd
                context_embd = mention.document().our_context_embd[mention_text]
                entity_embd = self._w2v.get_entity_vec(entity)
                if entity_embd is not None:
                    print self._w2v.similarity(context_embd, entity_embd)
                    features.append(self._w2v.similarity(context_embd, entity_embd))
                else:
                    print 0
                    features.append(0.0)
            elif feature.startswith('model_'):
                x = self.models_as_features_predictors[feature[6:]].predict_prob(mention, entity)
                features.append(x)
            else:
                raise "feature undefined"

        return features