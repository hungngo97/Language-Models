from utils.io_utils import (read_sentences_from_file)
from utils.math_utils import *
from constants.model_constants import (UNK, START, STOP)
from constants.constants import (DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.UnigramLanguageModel import (UnigramLanguageModel)
from models.BigramLanguageModel import (BigramLanguageModel)
from models.TrigramLanguageModel import (TrigramLanguageModel)
from models.LinearInterpolationLanguageModel import (LinearInterpolationLanguageModel)
from evaluator.evaluator import ( PerplexityEvaluator )

def print_perplexity_score(perplexity_evaluator, sentences, unigram_model = None, bigram_model = None, trigram_model = None):
    if unigram_model != None:
        print("Unigram Perplexity Score")
        print(perplexity_evaluator.get_unigram_perplexity(unigram_model, sentences))
    if bigram_model != None:
        print("Bigram Perplexity Score")
        print(perplexity_evaluator.get_bigram_perplexity(bigram_model, sentences))
    if trigram_model != None:
        print("Trigram Perplexity Score")
        print(perplexity_evaluator.get_trigram_perplexity(trigram_model, sentences))
        
#Read data input
sentences = read_sentences_from_file(TRAIN_DATA_FILE)
dev_sentences = read_sentences_from_file(DEV_DATA_FILE)
test_sentences = read_sentences_from_file(TEST_DATA_FILE)

# Training models
unigram_model = UnigramLanguageModel(sentences, K_smoothing=0)
unigram_model.calculate_unigram_probablities("the")
unigram_model.calculate_unigram_probablities("a")
unigram_model.calculate_unigram_probablities("dog")

bigram_model = BigramLanguageModel(sentences, k_smoothing=0)
bigram_model.calculate_bigram_probability("the", "car")
bigram_model.calculate_bigram_probability("the", "dog")
bigram_model.calculate_bigram_probability("the", "boy")


trigram_model = TrigramLanguageModel(sentences, k_smoothing=0)
trigram_model.calculate_trigram_probability("the", "walking" , "car")


linear_model = LinearInterpolationLanguageModel(sentences, 0.3, 0.5, 0.2, 0.5, 0.5, k_smoothing=0)
# Evaluation
print("***************** Without Smoothing **************")
print("========== Train set score evaluation =======")
perplexity_evaluator = PerplexityEvaluator()
print_perplexity_score(
        perplexity_evaluator,sentences,
        unigram_model, bigram_model, trigram_model
        )

print("========== Dev set score evaluation =======")
print_perplexity_score(
        perplexity_evaluator,dev_sentences,
        unigram_model, bigram_model, trigram_model
        )

print("========== Test set score evaluation =======")
print_perplexity_score(
        perplexity_evaluator,test_sentences,
        unigram_model, bigram_model, trigram_model
        )

print("******************* With Smoothing ****************")
k = 1 # Dev should be 5000, test 8000
unigram_model_smooth = UnigramLanguageModel(sentences, K_smoothing=k)
bigram_model_smooth = BigramLanguageModel(sentences, k_smoothing=k)
trigram_model_smooth = TrigramLanguageModel(sentences, k_smoothing=k)
print("============== K = " + str(k) + " ====================")
print("========== Train set score evaluation =======")
perplexity_evaluator = PerplexityEvaluator()
print_perplexity_score(
        perplexity_evaluator,sentences,
        unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
        )

print("========== Dev set score evaluation =======")
print_perplexity_score(
        perplexity_evaluator,dev_sentences,
        unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
        )

print("========== Test set score evaluation =======")
print_perplexity_score(
        perplexity_evaluator,test_sentences,
        unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
        )
for k in range(1,2,2):
    unigram_model_smooth = UnigramLanguageModel(sentences, K_smoothing=k)
    bigram_model_smooth = BigramLanguageModel(sentences, k_smoothing=k)
    trigram_model_smooth = TrigramLanguageModel(sentences, k_smoothing=k)
    print("============== K = " + str(k) + " ====================")
    print("========== Train set score evaluation =======")
    perplexity_evaluator = PerplexityEvaluator()
    print_perplexity_score(
            perplexity_evaluator,sentences,
            unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
            )
    
    print("========== Dev set score evaluation =======")
    print_perplexity_score(
            perplexity_evaluator,dev_sentences,
            unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
            )
    
    print("========== Test set score evaluation =======")
    print_perplexity_score(
            perplexity_evaluator,test_sentences,
            unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
            )

print("******************* Linear Interpolation model ****************")
print("========== Train set score evaluation =======")
perplexity_evaluator = PerplexityEvaluator()
print(perplexity_evaluator.get_trigram_perplexity(linear_model, sentences))
print("========== Dev set score evaluation =======")
print(perplexity_evaluator.get_trigram_perplexity(linear_model, dev_sentences))

print("========== Test set score evaluation =======")
print(perplexity_evaluator.get_trigram_perplexity(linear_model, test_sentences))
