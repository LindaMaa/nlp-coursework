"""
Foundations of Natural Language Processing

Assignment 1

"""
from collections import defaultdict, Counter

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Import the Twitter corpus and LgramModel
from nltk_model import *  # See the README inside the nltk_model folder for more information

# Import the Twitter corpus and LgramModel
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy, tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy, ", ".join(tweet)))


def compute_accuracy(classifier, data):
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :type data: list(tuple(list(any), str))
    :param data: A list with tuples of the form (list with features, label)
    :rtype float
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f, data):
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :type extractor_f: (str, str, str, str) -> list(any)
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :type data: list(tuple(str))
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :rtype list(tuple(list(any), str))
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class, train_features, **kwargs):
        """

        :type classifier_class: a class object of nltk.classify.api.ClassifierI
        :param classifier_class: the kind of classifier we want to create an instance of.
        :type train_features: list(tuple(list(any), str))
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype dict(any, int)
        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype str
        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1 [7 marks]
def train_LM(corpus):

    # create a list of alphabetic tokens & convert to lower-case
    corpus_tokens = []
    for word in corpus.words():
        if word.isalpha():
            lowered_word = word.lower()
            corpus_tokens.append(lowered_word)

    # train a bigram letter LM, turn on padding, use default smoothing
    lm = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)
    return lm


# Question 2 [7 marks]
def tweet_ent(file_name, bigram_model):
    
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets
    '''
    cleaned_list_of_tweets=[]
    list_of_tweets = xtwc.sents(file_name)

    for item in list_of_tweets:
        list_items=[]

        # clean tweets, remove non-alphabetic tokens
        for w in item:
            if w.isalpha():
                ww=w.lower()
                list_items.append(ww)

        # remove any tweets with fewer than 5 tokens remaining
        if len(list_items)>=5:
            cleaned_list_of_tweets.append(list_items)

    list_of_tuples=[]
    for tweet in cleaned_list_of_tweets:
        sum_of_entropies=0
        for word in tweet:

            # compute the average word entropy for each tweet 
            entropy_val= bigram_model.entropy(word, pad_left=True, pad_right=True,
                        verbose=False, perItem=True) 

            # normalise by “sentence” length
            sum_of_entropies+=entropy_val/len(tweet)
        norm_entropy=sum_of_entropies
        list_of_tuples.append((norm_entropy,tweet))

    # return list sorted in ascending order of average word entropy 
    sorted_list_of_tuples = sorted(list_of_tuples, key=lambda tup: tup[0]) 
    return sorted_list_of_tuples


# Question 3 [8 marks]
def open_question_3():
    return inspect.cleandoc(
    """At the start of the list, there are tweets with the lowest entropy (around 2.5) which
    are the most similar to the words in Brown corpus (natural English). Towards the middle 
    and end, the tweets become more different from English sentences, and at the end of the 
    list, the tweets with the highest entropy (around 17.5) do not even contain English characters.
    Tweets in a foreign language that use the standard English characters have lower entropy than 
    tweets that use non-English symbols.""")[0:500]


# Question 4 [8 marks]
def open_question_4() -> str:
    return inspect.cleandoc("""
    1.Remove hashtags before removing non-alpha words - keep words containing #, they are a part of the sentence, e. #news
    2.Remove any words which contain non-English characters, e. 'нах'
    3.Use lemmatization or stemming which will fix misspelled words/with a letter missing/added, e. sickk
    4.Replace forms like "she's" by "she is" so the word does not lose its meaning after the apostrophe is removed
    5.Remove consecutive duplicated words or single character words e. ('in', 'in')""")[0:500]


# Question 5 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweets and their letter bigram entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average letter bigram entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)), list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             non-English tweets and entropies
    '''

    # remove the botttom 10% of tweets
    cut = int(len(list_of_tweets_and_entropies)*0.9)
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[:cut]

    # extract a list of the entropy values
    list_of_entropies=[]
    for item in list_of_ascii_tweets_and_entropies:
        list_of_entropies.append(item[0])

    # compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # limit used for filtering
    threshold = mean + standard_deviation

    list_of_not_English_tweets_and_entropies=[]

    # get a list of "probably not English" tweets, that is
    #  ascii tweets with an entropy > mean + standard_deviation
    for item in list_of_ascii_tweets_and_entropies:
        if item[0]>threshold:
            list_of_not_English_tweets_and_entropies.append(item)
    list_of_not_English_tweets_and_entropies=sorted(list_of_not_English_tweets_and_entropies, key=lambda tup: tup[0]) 

    # as required 2 statistics and 2 lists
    return  mean, standard_deviation, list_of_ascii_tweets_and_entropies,list_of_not_English_tweets_and_entropies


# Question 6 [15 marks]
def open_question_6():
    
    return inspect.cleandoc(
"""3 problems:
1)Are we considering spoken or written English? What English level does the speaker/writer have (as this might affect entropy)? Assume written text and a native speaker.
2)Is the text informal or formal? Assume a representative sample of both - articles, blogs, news, some formal personal letters.
3)What is a word in this experiment? We assume a string of alphanumeric UTF-8 characters, separated by white spaces, all letters lower case, no punctuation. 
Experiment:
The experiment will be performed on a set of representative texts - this will ensure the representation of a variety of words and lengths. Model each word 
W as a binary random variable. Calculate the entropy of W using the formula H(W)=−P(W=0)log2P(W=0)−P(W=1)log2P(W=1) where W=1 means W occurs in sentence S 
and W=0 means otherwise. Do this for all words, all sentences to collect as many independent trials as possible. Calculate the median entropy for a word - 
median instead of mean will exclude outliers.""")[:1000]


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 7 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data, alpha):
        """
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)



    @staticmethod
    def get_vocab(data):
        """
        Compute the set of all possible features from the (training) data.
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :rtype: set(any)
        :return: The set of all features used in the training data for all classes.
        """
        vocab_set = set()
        for sample in data:
            # create a set of unique features
            for feature in sample[0]:
                vocab_set.add(feature)
        return vocab_set

    @staticmethod
    def train(data, alpha, vocab):
        assert alpha >= 0.0

        # compute prior P(c) for all classes
        dict_priors={}
        for sample in data:
            label=sample[1]
            if label in dict_priors.keys():
                dict_priors[label]+=1
            else:
                dict_priors[label]=1

        # probabilities need to add to 1
        for k in dict_priors:
            dict_priors[k]=dict_priors[k]/len(data)

        
        dict_counts={}
        for k in dict_priors.keys():
            dict_counts[k]={}
       
        # count the number of times feature occurs in class k 
        for k in dict_priors.keys():
            # iterate through all features in every document
            for document in data:
                if (k==document[1]):
                    for feature in document[0]:
                        if feature in dict_counts[k].keys():
                            dict_counts[k][feature]+=1
                        else:
                            dict_counts[k][feature]=1
        
        # use Lidstone smoothing & compute denominator
        class_counts={}
        for k in dict_priors.keys():
            class_counts[k]=0

        for k in dict_priors.keys():
            for f in vocab:
                if f in dict_counts[k].keys():
                    class_counts[k]+=dict_counts[k][f]+alpha
                else: 
                    class_counts[k]+=alpha

         # P(f|c) - feature in dict
        for k in dict_priors.keys():
            for f in dict_counts[k].keys():
                dict_counts[k][f]=(dict_counts[k][f]+alpha)/class_counts[k]
        
        # P(f|c) - feature not in dict
        for k in dict_priors.keys():
            for f in vocab:
                if f not in dict_counts[k].keys():
                    dict_counts[k][f]=(alpha)/class_counts[k]
        
        return dict_priors, dict_counts

    
    
    def prob_classify(self, d):
        sum_prob=0
        probs_classes_features={}
        
        # set initial probabilities to 0
        for c in self.likelihood.keys():
            probs_classes_features[c]=0

        # compute P(c|d) for all classes
        for feature in d:
            for c in self.likelihood.keys():
                if feature in self.likelihood[c].keys():
                    sum_prob+=self.likelihood[c][feature]*self.prior[c]
                    probs_classes_features[c]+=self.likelihood[c][feature]*self.prior[c]
        
        # "normalize" probabilities
        for c in probs_classes_features.keys():
            if (sum_prob>0):
                probs_classes_features[c]=probs_classes_features[c]/sum_prob

        return probs_classes_features


    def classify(self, d):
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :type d: list(any)
        :param d: A list of features.
        :rtype: str
        :return: The most likely class.
        """
        class_result = self.prob_classify(d)
        max=0
        max_k=None

        # return class with maximum probability for given document 
        for k in class_result.keys():
            if (class_result[k]>=max):
                max=class_result[k]
                max_k=k
        return max_k
                

# Question 8 [10 marks]
def open_question_8() -> str:
    """
    How do you interpret the differences in accuracy between the different ways to extract features?
    :rtype: str
    :return: Your answer of 500 characters maximum.
    """
    return inspect.cleandoc(
    """Extracting V, N1, or N2 alone has low accuracy. If extracting only 1 feature, P shows better performance.
    The last method is the best performing because it considers multiple features. The task is better solved if 
    we consider the features in combination because they provide more information. The LR model performs better 
    because NB assumes independence of each feature and therefore it has a higher bias and might never achieve 
    as high accuracy as LR given that the dataset is large enough.""")[:500]


# Feature extractors used in the table:
# see your_feature_extractor for documentation on arguments and types.
def feature_extractor_1(v, n1, p, n2):
    return [v]


def feature_extractor_2(v, n1, p, n2):
    return [n1]


def feature_extractor_3(v, n1, p, n2):
    return [p]


def feature_extractor_4(v, n1, p, n2):
    return [n2]


def feature_extractor_5(v, n1, p, n2):
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 9.1 [5 marks] 
# my feature templates differentiate based on most important prepositional phrases
# if n2 is numeric and has length 4, it is year
# if n2 is non-alpha, create a feaure
# check word endings, especially verbs
# stem each verb
# convert everything to lowercase

def your_feature_extractor(v, n1, p, n2):
    wnl = WordNetLemmatizer()
    ps =PorterStemmer()
    
    if (p=="of"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("of", p) ]
    
    if (p=="for"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("for", p)]
    
    if (p=="on"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("on", p)]
    
    if (p=="into"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("into", p)]
    
    if (p=="without"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("without", p)]
    
    if (p=="amid"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("amid", p)]
    
    if (p=="before"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("before", p)]

    if (p=="until"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("until", p)]
    
    if (p=="via"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("via", p)]

    if (p=="through"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("through", p)]
    
    if (p=="per"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("per", p)]
    
    if (p=="by"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("by", p)]
    
    if (p=="than"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("than", p)]
    
    if (p=="with"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("with", p)]
    
    if (p=="during"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("during", p)]
    
    if (p=="next"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("next", p)]
    
    if (p=="upon"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("upon", p)]
    
    
    if (n2.isalpha()):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("alpha", n2)]

    if (not(n2.isalpha()) and len(n2)==4):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("year", n2)]
    
    if not(n2.isalpha()):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("num", n2)]
   
    if (not(n1.isalpha()) and len(n2)==4):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("year_n1", n1)]

    if not(n1.isalpha()):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("num_n1", n1)]
    
    if (n1=="it"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("it", n1.lower())]
    
    l = len(n2)
    if n2[l - 3:]=='ing' or n2[l - 3:]=='not' or n2[l - 3:]=='ed':
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("action", n2)]
    l = len(v)
    if v[l - 3:]=='ing':
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("ing", ps.stem(v.lower()))]
    if v[l - 2:]=='ed':
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("ed", ps.stem(v.lower()))]
    li=len(n1)
    if (n1[li - 3:]=="ity"):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("ity", n1) ]
    
    if ("-" in n2):
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower()),("dash", n2) ]
        
    else:
        return [("v", ps.stem(v.lower())), ("n1", n1.lower()), ("p", p.lower()), ("n2", n2.lower())]
    

# Question 9.2 [10 marks]
def open_question_9():
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick 3 examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.
    :rtype: str
    :return: Your answer of 1000 characters maximum.
    """
    return inspect.cleandoc(
    """Feature templates include filtering for frequent values of p, e.g. "of", "for", "via" which are more represented in one class. 
    All extracted features are lower-cased and all verbs are converted to stem form.It is checked whether n2 is numeric and whether 
    it has length 4 (the number is likely year), if n2 is numeric it is more likely to be N class. Endings of verbs "ing","ed" are 
    checked and appropriate features are added. It is checked whether n2 contains "-" or whether it ends with "ing" or "ed".
    Informative features: 1 p=of, 2 p=without, 3 p=via
    These prepositions can help to discriminate between classes since they are more frequently occurring in V or NP attachment sentences 
    (in English not only in training data).There are other features that have larger weights like 'shambles' or 'luxury' but are NOT 
    informative, they just happen to occur a lot in one class or never occur in the other class - the model thinks they are important 
    but they do not help to resolve PP attachment.""")[:1000]


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""


def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_4, answer_open_question_3, answer_open_question_6,\
        answer_open_question_8, answer_open_question_9
    global ascci_ents, non_eng_ents

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features

    print("*** Part I***\n")

    print("*** Question 1 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 2 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)

    print("*** Question 3 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    print("*** Question 4 ***")
    answer_open_question_4 = open_question_4()
    print(answer_open_question_4)

    print("*** Question 5 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Tweets considered non-English')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

    print("*** Question 6 ***")
    answer_open_question_6 = open_question_6()
    print(answer_open_question_6)

    print("*** Part II***\n")

    print("*** Question 7 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")
 

    print("*** Question 8 ***")
    answer_open_question_8 = open_question_8()
    print(answer_open_question_8)

    # This is the code that generated the results in the table of the CW:

    # A single iteration of suffices for logistic regression for the simple feature extractors.
    #
    # extractors_and_iterations = [feature_extractor_1, feature_extractor_2, feature_extractor_3, eature_extractor_4, feature_extractor_5]
    #
    # print("Extractor    |  Accuracy")
    # print("------------------------")
    #
    # for i, ex_f in enumerate(extractors, start=1):
    #     training_features = apply_extractor(ex_f, ppattach.tuples("training"))
    #     dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    #
    #     a_logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=6, trace=0)
    #     lr_acc = compute_accuracy(a_logistic_regression_model, dev_features)
    #     print(f"Extractor {i}  |  {lr_acc*100}")


    print("*** Question 9 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_9 = open_question_9()
    print("Answer to open question:")
    print(answer_open_question_9)





if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
