from utils.io_utils import (read_sentences_from_file)
from utils.math_utils import *
from constants.model_constants import (UNK, START, STOP)
from constants.constants import (DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.UnigramLanguageModel import (UnigramLanguageModel)
from models.BigramLanguageModel import (BigramLanguageModel)
from evaluator.evaluator import ( PerplexityEvaluator )


sentences = read_sentences_from_file(TRAIN_DATA_FILE)

unigram_model = UnigramLanguageModel(sentences, K_smoothing=0)
unigram_model.calculate_unigram_probablities("the")
unigram_model.calculate_unigram_probablities("a")
unigram_model.calculate_unigram_probablities("dog")


bigram_model = BigramLanguageModel(sentences, k_smoothing=0)
bigram_model.calculate_bigram_probability("the", "car")
bigram_model.calculate_bigram_probability("the", "dog")
bigram_model.calculate_bigram_probability("the", "boy")

perplexity_evaluator = PerplexityEvaluator()
perplexity_evaluator.get_unigram_perplexity(unigram_model, sentences)

perplexity_evaluator.get_bigram_perplexity(bigram_model, sentences)
