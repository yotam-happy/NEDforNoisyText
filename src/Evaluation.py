import numpy as np
import time
import utils.text
import json


class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, model, candidator, stats=None,
                 sampling=None, log_path=None, db=None, trained_mentions=None, attn=0.0,
                 entity_transform=None):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._iter = test_iter
        self._model = model
        self._candidator = candidator
        self._sampling = sampling
        self._stats = stats
        self._log_path = log_path
        self._db = db
        self._trained_mentions = trained_mentions
        self._attn = attn
        self.entity_transform = entity_transform

        self.error_from_untrained_mention = 0

        self.n_samples = 0
        self.correct = 0
        self.possible = 0
        self.n_docs = 0
        self.macro_p = 0
        self.candidates = 0
        self.mps_correct = 0
        self.n_docs_for_macro = 0
        self.single_cand = 0

    def evaluate(self):
        """
        Do the work - runs over the given test/evaluation set and compares
        the predictions of the model to the actual sense.

        Populates the members:
        self.n_samples:     number of samples we tested on
        self.correct:       number of correct predictions
        self.no_prediction: number of samples the model did not return any prediction for

        :return:
        """
        self.single_cand = 0
        self.n_samples = 0
        self.n_docs = 0
        self.n_docs_for_macro = 0
        self.correct = 0
        self.mps_correct = 0
        self.possible = 0
        self.macro_p = 0
        self.candidates = 0
        _sampling_k = 0
        _sampling_j = 0

        self.acc_by_mention = dict()
        self.acc_by_sense = dict()

        predictor = self._model.getPredictor()

        for doc in self._iter.documents():
            self._candidator.add_candidates_to_document(doc)
            self.n_docs += 1

            correct_per_doc = 0
            possible_per_doc = 0
            for mention in doc.mentions:
                # deterministic sampling so we get the same subset every time
                _sampling_k += 1
                if self._sampling is not None and float(_sampling_j) / _sampling_k > self._sampling:
                    continue
                _sampling_j += 1

                self.n_samples += 1
                actual = mention.gold_sense_id()
                candidates = mention.candidates
                if self.entity_transform is not None:
                    actual = self.entity_transform.entity_id_transform(actual)

                if actual in candidates:
                    mps = self._stats.getMostProbableSense(mention)
                    if mps == actual:
                        self.mps_correct += 1

                    # some printing
#                    print mention.left_context(), '>>', mention.mention_text(), '<<', mention.right_context()
#                    print 'entity"', self._db.getPageTitle(mention.gold_sense_id())
#                    print 'actual:', self.entity_transform.id2string(actual)
#                    print 'mps:', self.entity_transform.id2string(mps)
#                    print 'candidates', {c: self.entity_transform.id2string(c) for c in candidates}

                    self.candidates += len(mention.candidates)
                    possible_per_doc += 1
                    prediction = predictor.predict(mention)
                    if len(mention.candidates) == 1:
                        self.single_cand += 1
#                    print prediction, actual
                    if prediction == actual:
#                        print "correct!"
                        correct_per_doc += 1

                        if self._trained_mentions is not None and mention.mention_text() not in self._trained_mentions:
                            self.error_from_untrained_mention += 1
                    if np.random.rand() < self._attn:
                        self.output_attn(mention)

            if self.n_docs % 10 == 0:
                self.printEvaluation()
            self.possible += possible_per_doc
            self.correct += correct_per_doc
            if possible_per_doc > 0:
                self.n_docs_for_macro += 1
                self.macro_p += float(correct_per_doc) / possible_per_doc

        print 'done!'
        self.printEvaluation()
        if self._log_path is not None:
            self.saveEvaluation()

    def output_attn(self, mention):
        attn_sum_left = None
        attn_sum_right = None
        k = 0
        for candidate in mention.candidates:
            ret = self._model.get_attn(mention, candidate, None)
            if ret is None:
                continue
            left_context, left_attn, right_context, right_attn = ret
            k += 1
            if attn_sum_left is None:
                attn_sum_left = np.array(left_attn)
                attn_sum_right = np.array(right_attn)
            else:
                attn_sum_left += np.array(left_attn)
                attn_sum_right += np.array(right_attn)

        if left_context is not None:
            attn_sum_left = attn_sum_left / k
            attn_sum_right = attn_sum_right / k

            left_context.reverse()
            left_attn.reverse()
            s = ''
            for i, w in enumerate(left_context):
                s += w + ' '
                if left_attn[i] > 0:
                    s += '(' + str(attn_sum_left[i]) + ') '
            s += mention.mention_text() + ' '
            for i, w in enumerate(right_context):
                s += w + ' '
                if right_attn[i] > 0:
                    s += '(' + str(attn_sum_right[i]) + ') '
            print 'attention:' + s + '\n'

    def mircoP(self):
        return float(self.correct) / self.possible if self.possible > 0 else 'n/a'
    def macroP(self):
        return self.macro_p / self.n_docs_for_macro if self.n_docs_for_macro > 0 else 'n/a'
    def printEvaluation(self):
        print self.evaluation()

    def saveEvaluation(self):
        with open(self._log_path, "a") as f:
            f.write(self.evaluation(advanced=True))
        print self.evaluation()

    def evaluation(self, advanced=False):
        """
        Pretty print results of evaluation
        """
        tried = float(self.possible) / self.n_samples if self.n_samples > 0 else 'n/a'
        avg_cands = float(self.candidates) / self.possible if self.possible > 0 else 'n/a'
        micro_p = self.mircoP()
        macro_p = self.macroP()
        mps_correct = float(self.mps_correct) / self.possible if self.possible > 0 else 'n/a'
        on = float(self.single_cand) / self.possible
        s = time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + ": " + \
            str(self.n_samples) + ' samples in ' + str(self.n_docs) + ' docs. ' + str(tried) + \
            '% mentions tried, avg. candidates per mention:", ' + str(avg_cands) + \
            ' ". micro p@1: ' + str(micro_p) + '% macro p@1: ' + str(macro_p) + "% mps p@1: " + str(mps_correct) + '%\n'
        return s
