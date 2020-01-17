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
            bigram_count += len(sentence) - 1 #Ignore the 1st (None, START)
        return bigram_count
    
    def get_total_trigram(self, sentences):
        trigram_count = 0 
        for sentence in sentences:
            trigram_count += len(sentence) - 2#Ignore the (None, None, Start) & (None, START, W_i)
        return trigram_count
    
    def get_unigram_perplexity(self, model, sentences):
        unigram_count = self.get_total_unigram(sentences)
        sentence_prob_log_sum = 0
        for sentence in sentences:
            sentence_prob_log_sum += model.calculate_sentence_log_probability(sentence)
            """try:
                sentence_prob_log_sum += model.calculate_sentence_log_probability(sentence)
            except:
                # If met a sentence that we haven't seen before, assign infinity
                sentence_prob_log_sum += math.log(2e-20, 2)"""
        return math.pow(2, -1 * sentence_prob_log_sum / unigram_count)
    
    def get_bigram_perplexity(self, model, sentences):
        bigram_count = self.get_total_bigram(sentences)
        bigram_prob_log_sum = 0
        for sentence in sentences:
            bigram_prob_log_sum += model.calculate_bigram_sentence_log_probability(sentence)
            """
            try:
                bigram_prob_log_sum += model.calculate_bigram_sentence_log_probability(sentence)
            except:
                # If met a sentence that we haven't seen before, assign infinity
                bigram_prob_log_sum += float('-inf')
            """
        return math.pow(2, -1 * bigram_prob_log_sum / bigram_count)
    
    def get_trigram_perplexity(self, model, sentences):
        trigram_count = self.get_total_trigram(sentences)
        trigram_prob_log_sum = 0
        for sentence in sentences:
            trigram_prob_log_sum += model.calculate_trigram_sentence_log_probability(sentence)
            """
            try:
                trigram_prob_log_sum += model.calculate_trigram_sentence_log_probability(sentence)
            except:
                # If met a sentence that we haven't seen before, assign infinity
                trigram_prob_log_sum += float('-inf')
            """
        return math.pow(2, -1 * trigram_prob_log_sum / trigram_count)
    
