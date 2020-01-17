# -*- coding: utf-8 -*-
"""

"""
import math
class PerplexityEvaluator:
    def __init__(self):
        self.name = "Perplexity Evaluator"
        
    def get_total_unigram(self, sentences):
        unigram_count = 0
        for sentence in sentences:
            unigram_count += len(sentence) - 2 #Not counting START and END
            #TODO: Fix this for UNK
        return unigram_count
    
    
    def get_total_bigram(self, sentences):
        bigram_count = 0
        for sentence in sentences:
            bigram_count += len(sentence) - 1 #TODO: Ignore the 1st (None, START)
        return bigram_count
    
    def get_unigram_perplexity(self, model, sentences):
        unigram_count = self.get_total_unigram(sentences)
        sentence_prob_log_sum = 0
        for sentence in sentences:
            sentence_prob_log_sum += model.calculate_sentence_log_probability(sentence)
            
            #TODO: Not sure if needed to catch an invalid case
        return math.pow(2, sentence_prob_log_sum / unigram_count)
    
    def get_bigram_perplexity(self, model, sentences):
        bigram_count = self.get_total_bigram(sentences)
        bigram_prob_log_sum = 0
        for sentence in sentences:
            bigram_prob_log_sum += model.calculate_bigram_sentence_log_probability(sentence)
        return math.pow(2, bigram_prob_log_sum / bigram_count)

