# -*- coding: utf-8 -*-
import re
from constants.model_constants import (STOP, START)
"""
    Read sentences from a file and split all string into a list of words in each
    string
    
    @Returns: List of list of words with each inner list is a sentence
"""
def read_sentences_from_file(file_path):
    with open(file_path, "r") as file:
        sentences = []
        for sentence in file:
            words = [START]
            words = words + re.split("\s+", sentence.rstrip())
            words = words + [STOP]
            sentences.append(words)
        return sentences
    
    
