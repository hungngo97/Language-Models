from constants.model_constants import (UNK, START, STOP)
import math
from .UnigramLanguageModel import UnigramLanguageModel

class BigramLanguageModel(UnigramLanguageModel):
    """
        @Param: Sentences with each sentence is a list of words
        @Param smoothing: Function that do smoothing
    """
    def __init__(self, sentences, k_smoothing=0):
        UnigramLanguageModel.__init__(self, sentences, k_smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in sentences:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word), 0) + 1
                    if previous_word != START and word != STOP:
                        self.unique_bigrams.add((previous_word, word))
                previous_word = word
        self.total_unique_bigrams = len(self.bigram_frequencies)
        
    
    def calculate_bigram_probability(self, previous_word, word):
        numerator = self.bigram_frequencies.get((previous_word, word), 0)
        denominator = self.unigram_frequencies.get(previous_word, 0)
        # TODO: Add K smoothing here
        if self.k_smoothing:
            numerator += self.k_smoothing
            # TODO: Not sure if this smoothing is correct
            denominator += self.total_unique_bigrams + self.k_smoothing
        return 0.0 if denominator == 0 else float(numerator) / float(denominator)
    
    def calculate_bigram_sentence_log_probability(self, sentence):
        sentence_log_probability = 0
        previous_word = None
        for word in sentence:
            if previous_word != None:
                bigram_prob = self.calculate_bigram_probability(previous_word, word)
                sentence_log_probability += math.log(bigram_prob, 2)
            previous_word = word
        return sentence_log_probability