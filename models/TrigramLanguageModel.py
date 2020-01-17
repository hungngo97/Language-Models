# -*- coding: utf-8 -*-
from constants.model_constants import (UNK, START, STOP)
import math
from .UnigramLanguageModel import UnigramLanguageModel
from .BigramLanguageModel import BigramLanguageModel

class TrigramLanguageModel(BigramLanguageModel):
    """
        @Param: Sentences with each sentence is a list of words
        @Param smoothing: Function that do smoothing
    """
    def __init__(self, sentences, k_smoothing=0):
        BigramLanguageModel.__init__(self, sentences, k_smoothing)
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()
        for sentence in sentences:
            previous_previous_word = None 
            previous_word = None
            for word in sentence:
                if previous_word != None and previous_previous_word != None:
                    self.trigram_frequencies[(previous_previous_word, previous_word, word)] = \
                        self.trigram_frequencies.get((previous_previous_word, previous_word, word), 0) + 1
                    if previous_previous_word != START and previous_word != START and word != STOP:
                        self.unique_trigrams.add((previous_previous_word, previous_word, word))
                previous_previous_word = previous_word
                previous_word = word
        self.total_unique_trigrams = len(self.bigram_frequencies)
        
    
    def calculate_trigram_probability(self, previous_previous_word, previous_word, word):
        numerator = self.bigram_frequencies.get((previous_word, word), 0)
        denominator = self.unigram_frequencies.get(previous_word, 0)
        # TODO: Add K smoothing here
        if self.k_smoothing:
            numerator += self.k_smoothing
            # TODO: Not sure if this smoothing is correct
            denominator += self.total_unique_trigrams + self.k_smoothing
        return 0.0 if denominator == 0 else float(numerator) / float(denominator)
    
    def calculate_trigram_sentence_log_probability(self, sentence):
        sentence_log_probability = 0
        previous_previous_word = None
        previous_word = None
        for word in sentence:
            if previous_previous_word != None and previous_word != None:
                trigram_prob = self.calculate_trigram_probability(
                        previous_previous_word, previous_word, word
                        )
                sentence_log_probability += math.log(trigram_prob, 2)
            previous_previous_word = previous_word
            previous_word = word
        return sentence_log_probability
