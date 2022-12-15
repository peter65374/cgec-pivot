import nltk
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize.casual import reduce_lengthening
from math import log
import csv
'''
To finds the top 100 word pairs that appear significantly more often in the same documents in a given collection 
of tweets(every line is a tweet) using pointwise mutual information, mutual information, and Chi-square (χ2). 
Result: The results of pointwise mutual information and Chi-square are almost the same, mutual information 
differs from them. Mutual information method is more suitable for this task.
'''

def main():
    # read file rather than line can cause problems with splitting words at the end of each line with the first of the next line
    file = open('tweets.txt', 'r')
    unigram_counter = Counter()
    bigram_counter = Counter()
    N = 0
    for line in file:
        N += 1
        line = str(line)
        clean_words = basic_processing(line)
        stemmed_words = stem_words(clean_words)
        bigrams = nltk.bigrams(stemmed_words)
        for unigram in set(stemmed_words):
            unigram_counter[unigram] += 1
        for bigram in set(bigrams):
            bigram_counter[bigram] += 1
    '''
    # word collocations doesn't need to consider sequence, combine the two keys in different order
    for (word1, word2) in bigram_counter.copy():
        bigram_counter[(word1, word2)] += bigram_counter.get((word2, word1), 0)
        if bigram_counter.get((word2, word1)):
            del bigram_counter[(word2, word1)]
    '''
    print(pointwise_mutual_information(unigram_counter, bigram_counter, N)[0:100])
    print(mutual_information(unigram_counter, bigram_counter, N)[0:100])
    print(chi_square(unigram_counter, bigram_counter, N)[0:100])


def pointwise_mutual_information(unigram_counter, bigram_counter, N):
    all_pmi = list()
    for word1, word2 in bigram_counter:
        word1_counts = unigram_counter[word1]
        word2_counts = unigram_counter[word2]
        word1_word2_counts = bigram_counter[(word1, word2)]
        pmi = log((word1_word2_counts / N) / ((word1_counts / N) * (word2_counts / N)), 2)
        all_pmi.append([(word1, word2), pmi, word1_word2_counts, word1, word1_counts, word2, word2_counts])
    sorted_pmi = sorted(all_pmi, key=lambda elements: elements[1], reverse=True)
    with open('pmi.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(["Bigram", "Pointwise Mutual Information", " Bigram Counts", "Word1", "Word1 Counts", "Word2", "Word2 Counts"])
        for i in range(0, 99):
            w.writerow(sorted_pmi[i])
    return sorted_pmi


def mutual_information(unigram_counter, bigram_counter, N):
    all_mi = list()
    for word1, word2 in bigram_counter:
        word1_counts = unigram_counter[word1]
        word2_counts = unigram_counter[word2]

        word1_word2_counts = bigram_counter[(word1, word2)]
        word1_no_word2_counts = unigram_counter[word1] - bigram_counter[(word1, word2)]
        word2_no_word1_counts = unigram_counter[word2] - bigram_counter[(word1, word2)]
        no_word1_no_word2_counts = N - word1_word2_counts - word1_no_word2_counts - word2_no_word1_counts

        try:
            prob_word1_word2 = (word1_word2_counts / N) * log(N * word1_word2_counts / (word1_counts * word2_counts), 2)
        except:
            prob_word1_word2 = 0
        try:
            prob_word1_no_word2 = (word1_no_word2_counts / N) * log(N * word1_no_word2_counts / (word1_counts * (N - word2_counts)), 2)
        except:
            prob_word1_no_word2 = 0
        try:
            prob_word2_no_word1 = (word2_no_word1_counts / N) * log(N * word2_no_word1_counts / (word2_counts * (N - word1_counts)), 2)
        except:
            prob_word2_no_word1 = 0
        try:
            prob_no_word1_no_word2 = (no_word1_no_word2_counts / N) * log(N * no_word1_no_word2_counts / ((N - word1_counts) * (N - word2_counts)), 2)
        except:
            prob_no_word1_no_word2 = 0
        mi = prob_word1_word2 + prob_word1_no_word2 + prob_word2_no_word1 + prob_no_word1_no_word2
        all_mi.append([(word1, word2), mi, word1_word2_counts, word1, word1_counts, word2, word2_counts])
    sorted_mi = sorted(all_mi, key=lambda elements: elements[1], reverse=True)
    with open('mi.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(["Bigram", "Mutual Information", " Bigram Counts", "Word1", "Word1 Counts", "Word2", "Word2 Counts"])
        for i in range(0, 99):
            w.writerow(sorted_mi[i])
    return sorted_mi


def chi_square(unigram_counter, bigram_counter, N):
    all_chi = list()
    for word1, word2 in bigram_counter:
        word1_counts = unigram_counter[word1]
        word2_counts = unigram_counter[word2]

        word1_word2_counts = bigram_counter[(word1, word2)]
        word1_no_word2_counts = unigram_counter[word1] - bigram_counter[(word1, word2)]
        word2_no_word1_counts = unigram_counter[word2] - bigram_counter[(word1, word2)]
        no_word1_no_word2_counts = N - word1_word2_counts - word1_no_word2_counts - word2_no_word1_counts

        numerator = (N * pow((word1_word2_counts * no_word1_no_word2_counts - word1_no_word2_counts * word2_no_word1_counts), 2))
        denominator = (word1_word2_counts + word2_no_word1_counts) * (word1_word2_counts + word1_no_word2_counts) * (word1_no_word2_counts + no_word1_no_word2_counts) * (word2_no_word1_counts + no_word1_no_word2_counts)
        chi = numerator/denominator
        all_chi.append([(word1, word2), chi, word1_word2_counts, word1, word1_counts, word2, word2_counts])
    sorted_chi = sorted(all_chi, key=lambda elements: elements[1], reverse=True)
    with open('chi.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(["Bigram", "Chi Square Score", " Bigram Counts", "Word1", "Word1 Counts", "Word2", "Word2 Counts"])
        for i in range(0, 99):
            w.writerow(sorted_chi[i])
    return sorted_chi


def basic_processing(line):
    # get rid of certain type: @username and #hashtag
    clean_texts = re.sub(r'[@#][\w_]*', '', line)
    # get rid of chars other than a-zA-Z0-9' '
    words = re.sub(r'[^\w ]+', '', clean_texts).lower().split()
    '''
    # replace repeated character sequences of length 3 or greater
    for word in words:
        pattern = re.compile(r"(.)\1{3,}")
        pattern.sub(r"\1", word)
    '''
    clean_words = [word for word in words if word.isalpha() and word not in stopwords.words('english', 'spanish')]
    return clean_words


def stem_words(line):
    wnl = WordNetLemmatizer()
    # stem verbs
    verb_stem = []
    for word in line:
        verb_stem.append(wnl.lemmatize(word, 'v'))
    # stem verbs
    stemmed_words = []
    for word in verb_stem:
        stemmed_words.append(wnl.lemmatize(word, 'n'))
    return stemmed_words


main()