import nltk
import re
import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
from colorama import Fore, Back, Style

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pickle
from collections import defaultdict

import spacy

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

nltk.download('stopwords')

from nltk.corpus import stopwords

nlp = spacy.load('en', disable=['parser', 'ner'])

class LDA:
    # NLTK stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    df = None
    data = None
    data_words = None
    bigram = None
    trigram = None
    bigram_mod = None
    trigram_mod = None
    fine_data = None
    id2word = None
    corpus = None
    lda_model = None
    doc_lda = None

    def __init__(self):
        self.load_data()
        self.create_bigram_and_trigram()
        self.fine_preprocess()
        self.create_dict_and_corpus()
        print(len(self.corpus))
        self.build_lda_model()

    # Load Data
    # Currently loading experiment data
    def load_data(self):
        print(Fore.BLUE + "************LOADING DATA************")
        # Import Dataset
        # self.df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
        # self.df.head()
        #
        # Clean data
        import pickle as pkl
        self.data = []
        df = pkl.load(open("/Users/longchen/Desktop/DS_1011_project/df_lda.pkl", 'rb'))

        self.data = df.tolist()

        # self.data = self.df.content.values.tolist()
        # self.remove_noise()

        # Tokenize data
        self.data_words = list(self.sent_to_words(self.data))
        del self.data

        print(Fore.GREEN + "\n********LOADING DATA COMPLETE*******")

    # Remove emails, newline, extra spaces, distracting single quote and urls
    def remove_noise(self):
        # Remove Emails
        self.data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in self.data]

        # Remove new line characters
        self.data = [re.sub(r'\s+', ' ', sent) for sent in self.data]

        # Remove distracting single quotes
        self.data = [re.sub("\'", "", sent) for sent in self.data]

        # Remove url
        self.data = [re.sub(r"http\S+", "", sent) for sent in self.data]

        self.data = self.mask(self.data)

    # Tokenize data
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    # Create bigram and trigram model
    def create_bigram_and_trigram(self, min_count=5, threshold=100):
        print(Fore.BLUE + "\n******Creating bigram and trigram******")
        self.bigram = gensim.models.Phrases(self.data_words, min_count=min_count, threshold=threshold)
        self.trigram = gensim.models.Phrases(self.bigram[self.data_words], threshold=threshold)

        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)

        del self.bigram
        del self.trigram

        print(Fore.GREEN + "\n******Finished creating bigram and trigram******\n")

    # Remove stopwords, using simple_preprocess in gensim, with stopwords from NLTK
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in texts]

    # Make bigram for the entire dataset
    def make_bigram(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    # Make trigram for the entire dataset
    def make_trigram(self, texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    # Lemmatize dataset, using spaCy
    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        ret_text = []
        for sent in texts:
            temp = nlp(" ".join(sent))
            ret_text.append([token.lemma_ for token in temp if token.pos_ in allowed_postags])

        return ret_text

    def mask(self, texts):
        word_list = []
        ret_text = []
        for text in texts:
            text = text.lower()
            for i in word_list:
                text = text.replace(i, ' ')
            ret_text.append(text)

        return ret_text

    # Remove stopwords, bigram, trigram and lemmatization
    def fine_preprocess(self):
        print(Fore.BLUE + "******Start fine preprocess******")

        self.fine_data = self.remove_stopwords(self.data_words)  # Remove stop words
        self.fine_data = self.make_bigram(self.fine_data)  # Form bigrams
        self.fine_data = self.lemmatization(self.fine_data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        print(Fore.GREEN + "******Finished fine preprocess******\n")

    # Create dictionary and corpus as LDA input
    def create_dict_and_corpus(self):
        print(Fore.BLUE + "******Start creating dictionary and corpus for LDA******")
        # Create dictionary
        self.id2word = corpora.Dictionary(self.fine_data)
        self.id2word.save('id2word.gensim')

        # Create corpus and term document frequency
        self.corpus = [self.id2word.doc2bow(text) for text in self.fine_data]

        print(Fore.GREEN + "******Finished creating dictionary and corpus for LDA******\n")

    # Build LDA model, print perplexity and coherence score. Currently using Mallet
    def build_lda_model(self):
        print(Fore.BLUE + "******Start building LDA model******\n")
        # coherence_dict = defaultdict(float)
        pickle.dump(self.corpus, open('corpus.pkl', 'wb'))
        coherence_values = []
        model_list = []
        for num_topics in range(3, 8):
            print("Number of topic is: ", num_topics)
            self.lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=self.corpus,
                                                                     id2word=self.id2word,
                                                                     num_topics=num_topics,
                                                                     workers=3,
                                                                     chunksize=200,
                                                                     passes=200,
                                                                     eval_every=1,
                                                                     per_word_topics=True)
            self.lda_model.save('lda_train_{}.model'.format(num_topics))

            # with open("LDA_result_" + str(num_topics), 'w') as output_file:
            #     for i in range(0, self.lda_model.num_topics):
            #         output_file.write(self.lda_model.print_topic(i) + '\n')

            model_list.append(self.lda_model)
            coherencemodel = CoherenceModel(model=self.lda_model, texts=self.fine_data, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        print(coherence_values)

        print(Fore.GREEN + "******Finished building LDA model******")

    # Get perplexity and coherence score
    def perplexity_and_coherence(self):
        # Compute Perplexity
        # print('\nPerplexity: ', self.lda_model.log_perplexity(self.corpus))

        # Compute Coherence Score
        coherence = CoherenceModel(model=self.lda_model, texts=self.fine_data, dictionary=self.id2word, coherence='c_v')
        print('\nCoherence Score: ', coherence.get_coherence())


if __name__ == '__main__':
    lda = LDA()
