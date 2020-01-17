# -*- coding: utf-8 -*-
import re
import random
from constants.model_constants import (UNK, STOP, START)
"""
    Read sentences from a file and split all string into a list of words in each
    string
    
    @Returns: List of list of words with each inner list is a sentence
"""

# TODO: Handle UNKing here
# TODO: Handle bigram, trigram, sentence haven't seen in test and dev
def read_sentences_from_file(file_path, unk=True, unk_threshold=3):
    with open(file_path, "r") as file:
        sentences = []
        for sentence in file:
            words = [START]
            words = words + re.split("\s+", sentence.rstrip())
            words = words + [STOP]
            sentences.append(words)
        if (unk):
            sentences = unk_sentences(sentences, unk_threshold=3, unk_prob=0.5)
        return sentences
    
def unk_sentences(sentences, unk_threshold=3, unk_prob=0.5):
    token_frequency = dict()
    """
    1) Count the frequency of all tokens in corpus.
    2) Choose a cutoff and some UNK probability (e.g. 5 and 50%)
    3) For all **individual tokens** that appear at or below cutoff, replace 50% of them with UNK.
    4) Estimate the probabilities for from its counts just like any other regular
    word in the training set.
    5) At dev/test time, replace words model hasn't seen before with UNK.
    """
    for sentence in sentences:
        for word in sentence:
            token_frequency[word] = token_frequency.get(word, 0) + 1
            
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if (token_frequency[word] < unk_threshold):
                # Replace the current token with UNK with UNK probability
                if (random.random() > unk_prob):
                    sentence[i] = UNK
    return sentences
                
        
        
