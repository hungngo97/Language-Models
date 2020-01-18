# -*- coding: utf-8 -*-
from constants.model_constants import (UNK, START, STOP, LOW_PROB)
import math
from .UnigramLanguageModel import UnigramLanguageModel
from .BigramLanguageModel import BigramLanguageModel
from .TrigramLanguageModel import TrigramLanguageModel

class LinearInterpolationLanguageModel(TrigramLanguageModel):
    """
        @Param: Sentences with each sentence is a list of words
        @Param smoothing: Function that do smoothing
    """
    def __init__(self, sentences, lambda1, lambda2, lambda3, lambda4, lambda5, k_smoothing=0):
        TrigramLanguageModel.__init__(self, sentences, k_smoothing)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        
    
    def calculate_linear_probability(self, previous_previous_word, previous_word, word):
        unigram_prob = self.calculate_unigram_probablities(word)
        bigram_prob = self.calculate_bigram_probability(previous_word, word)
        trigram_prob = self.calculate_trigram_probability(
                previous_previous_word, previous_word, word)
        if (trigram_prob == None or trigram_prob == None):
            print('Fall back error due to trigram 0')
            return self.lambda4 * unigram_prob + self.lambda5 * bigram_prob
        return self.lambda1 * unigram_prob + self.lambda2 * bigram_prob + self.lambda3 * trigram_prob
    
    def calculate_trigram_sentence_log_probability(self, sentence):
        sentence_log_probability = 0
        previous_previous_word = None
        previous_word = None
        for word in sentence:
            if previous_previous_word != None and previous_word != None:
                linear_prob = self.calculate_linear_probability(
                        previous_previous_word, previous_word, word
                        )
                if linear_prob == 0:
                    return float('-inf')
                sentence_log_probability += math.log(linear_prob, 2)
            previous_previous_word = previous_word
            previous_word = word
        return sentence_log_probability
