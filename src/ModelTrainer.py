from WikilinksStatistics import *
from Word2vecLoader import *
from bisect import bisect
import utils.text
from timeit import default_timer as timer



class ModelTrainer:
    """
    This class generates pairwise training examples from the corpus: (wikilink, candidate1, candidate2, correct)
    and feeds them to a model's train method
    """

    def __init__(self, iter, candidator, stats, model, epochs=10, neg_sample=1,
                 neg_sample_uniform=True, neg_sample_all_senses_prob=0.0,
                 subsampling=None, sampling=None, entity_transform=None):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._neg_sample = neg_sample
        self._iter = iter
        self._model = model
        self._stats = stats
        self._candidator = candidator
        self._epochs = epochs
        self.subsampling = subsampling
        self._sampling = sampling
        self.entity_transform = entity_transform

        self._n = 0
        self._nn = 0

        # calculate mean number of false candidates per mention
        tot_false_cand = 0
        tot_cases = 0
        for cand_list in self._stats.mentionLinks.itervalues():
            n_false_cands = len(cand_list) - 1
            n_cases = sum(cand_list.itervalues())
            tot_false_cand += n_false_cands * n_cases
            tot_cases += n_cases
        self.mean_false_cands_per_case = float(tot_false_cand) / tot_cases
        print 'mean candidates per case', self.mean_false_cands_per_case

        # setup all-sense negative-sampling (define cumulative probability function)
        # -- some ppl say that for long lists it is better to have small probs first due to precision issues
        senses = [(int(x), int(y)) for x, y in self._stats.conceptCounts.items()]

        senses = sorted(senses, key=lambda tup: tup[1])
        self._all_senses, self._all_senses_cpf = [e[0] for e in senses], [e[1] for e in senses]
        self._all_senses_cpf_total = 0
        for i in xrange(len(self._all_senses_cpf)):
            self._all_senses_cpf_total += self._all_senses_cpf[i]
            self._all_senses_cpf[i] = self._all_senses_cpf_total

        self._neg_sample_uniform = neg_sample_uniform
        self._neg_sample_all_senses_prob = neg_sample_all_senses_prob

    def getSenseNegSample(self):
        res = None
        while res is None:
            if self._neg_sample_uniform:
                res = self._all_senses[np.random.randint(len(self._all_senses))]
            else:
                x = np.random.randint(self._all_senses_cpf_total)
                i = bisect(self._all_senses_cpf, x)
                res = self._all_senses[i]
            if self.subsampling is not None:
                prior = self._stats.getCandidatePrior(res)
                discard_prob = 1.0 - math.sqrt(float(self.subsampling) / prior)
                if discard_prob > 0.0 and np.random.random() < discard_prob:
                    res = None
        return res

    def train_on_mention(self, mention):
        actual = mention.gold_sense_id()
        if self.entity_transform is not None:
            actual = self.entity_transform.entity_id_transform(actual)
        if actual is None:
            return

        self._n += 1
        if self.subsampling is not None:
            prior = self._stats.getCandidatePrior(actual)
            discard_prob = 1.0 - math.sqrt(float(self.subsampling) / prior)
            if discard_prob > 0.0 and np.random.random() < discard_prob:
                self._nn += 1
                return

        if self._n % 1000 == 0 and self._nn > 0:
            print "subsampled", float(self._nn) / self._n

        candidates = mention.candidates
        if actual not in candidates or not self._model.is_trainable(actual):
            return

        # get id vector
        ids = [candidate for candidate in candidates
               if int(candidate) != actual and self._model.is_trainable(candidate)]

        neg = []
        if type(self._neg_sample) is not int and self._neg_sample == 'all':
            neg = ids
        else:
            # get list of negative samples
            for k in xrange(self._neg_sample):
                # do negative sampling (get a negative sample)
                if np.random.rand() < self._neg_sample_all_senses_prob:
                    # get negative sample from all possible senses
                    neg.append(self.getSenseNegSample())
                else:
                    # get negative sample from senses seen for the current mention
                    if len(ids) < 1:
                        continue
                    neg.append(ids[np.random.randint(len(ids))])

        # train
        self._model.train(mention, actual, True)
        if len(neg) > 0:
            for wrong in neg:
                self._model.train(mention, wrong, False)

    def train(self):
        print "start training..."
        for epoch in xrange(self._epochs):
            print "training epoch ", epoch

            start_t = timer()
            k = 0
            for doc in self._iter.documents():
                self._candidator.add_candidates_to_document(doc)
                for mention in doc.mentions:
                    if self._sampling is not None and np.random.rand() > self._sampling:
                        continue
                    self.train_on_mention(mention)
                    k += 1
                    if k % 500 == 0:
#                        print "trained on", k
                        for_batches = 100 if len(self._model.train_loss) > 100 else len(self._model.train_loss)
                        print "trained on", k, "examples. avg. loss:", np.mean(self._model.train_loss[-for_batches:])
            print "epoch had", k, "mentions"
            end_t = timer()
            print "overall took - ", (end_t - start_t) / 60.0, " minutes for ", k, " mentions"

            if hasattr(self._model, "train_loss"):
                loss = sum(self._model.train_loss) / float(len(self._model.train_loss))
                print "avg. loss for epoch:", loss
        self._model.finalize()
        print "done training."
