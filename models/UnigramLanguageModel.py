from constants.model_constants import (UNK, START, STOP)
import math

class UnigramLanguageModel:
    """
        @Param: Sentences with each sentence is a list of words
        @Param smoothing: Function that do smoothing
    """
    def __init__(self, sentences, K_smoothing=0):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        self.k_smoothing = K_smoothing

        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != START and word != STOP:
                    self.corpus_length += 1
        self.unique_words = len(self.unigram_frequencies) - 2
                
    
    def calculate_unigram_probablities(self, word):
        if word not in self.unigram_frequencies:
            word = UNK
        word_probability_numerator = self.unigram_frequencies.get(word)
        word_probability_denominator = self.corpus_length
        if self.k_smoothing > 0:
            # TODO: Implemented add K Smoothing here?, currently it is add 1
            word_probability_numerator += self.k_smoothing
            word_probability_denominator += self.unique_words + self.k_smoothing
        return float(word_probability_numerator) / float(word_probability_denominator)
    
    
    def calculate_sentence_log_probability(self, sentence):
        sentence_probability_log_sum = 0
        for word in sentence:
            if word != START and word != STOP:
                word_probability = self.calculate_unigram_probablities(word)
                if (word_probability == 0):
                    return float('-inf')
                sentence_probability_log_sum += math.log(word_probability, 2)
        return sentence_probability_log_sum
    
    
    def generate_sentence(self, word):
        return "NOT_IMPLEMENTED"
        