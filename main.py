from utils.io_utils import (read_sentences_from_file)
from utils.math_utils import *
from constants.model_constants import (UNK, START, STOP)
from constants.constants import (DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.UnigramLanguageModel import (UnigramLanguageModel)

sentences = read_sentences_from_file(TRAIN_DATA_FILE)

unigram_model = UnigramLanguageModel(sentences, K_smoothing=0)
unigram_model.calculate_unigram_probablities("the")
unigram_model.calculate_unigram_probablities("a")
unigram_model.calculate_unigram_probablities("dog")