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
        self.trigram_count = 0
        for sentence in sentences:
            previous_previous_word = None 
            previous_word = None
            self.trigram_count += len(sentence) - 2
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
        if previous_previous_word not in self.unigram_frequencies:
            previous_previous_word = UNK
        if previous_word not in self.unigram_frequencies:
            previous_word = UNK
        if word not in self.unigram_frequencies:
            word = UNK
        numerator = self.trigram_frequencies.get((previous_previous_word, previous_word, word), 0)
        denominator = self.bigram_frequencies.get((previous_word, word), 0)
        if self.k_smoothing:
            numerator += self.k_smoothing
            denominator += self.unique_words * self.k_smoothing
            # denominator = self.unique_words * \
             #   (self.bigram_frequencies.get((previous_word, word), 0) + self.k_smoothing)
            # denominator = self.unique_words * \
              #  ( self.k_smoothing + self.trigram_count )
                
            
        if (denominator == 0):
            return 0
        return float(numerator) / float(denominator)
    
    def calculate_trigram_sentence_log_probability(self, sentence):
        sentence_log_probability = 0
        previous_previous_word = None
        previous_word = None
        for word in sentence:
            if previous_previous_word != None and previous_word != None:
                trigram_prob = self.calculate_trigram_probability(
                        previous_previous_word, previous_word, word
                        )
                if (trigram_prob == 0):
                    return float('-inf')
                sentence_log_probability += math.log(trigram_prob, 2)
            previous_previous_word = previous_word
            previous_word = word
        return sentence_log_probability
