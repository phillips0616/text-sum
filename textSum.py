__author__ = "phillips0616"

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

#determines the frequency of a each word in the article that isn't a stopword
def word_frequency(text, stopword_list):
    frequencies = {}
    word_sentences = [word_tokenize(sents.lower()) for sents in text]
    for entry in word_sentences:
        for word in entry:
            if word not in frequencies and word not in stopword_list:
                frequencies[word] = 1
            elif word in frequencies:
                frequencies[word] += 1
    frequencies = normalize_dict(frequencies,0.9,0.3)
    return frequencies

#noramlizes frequencies of words for ranking
#upperbound is used to cut out words used too often from ranking
#lowerbound is used to cut out words used too little from ranking
def normalize_dict(dictionary, upperbound, lowerbound):
    normalized_dict = {}
    max_value = max(dictionary.values())
    for entry in dictionary.keys():
        value = dictionary[entry] = float(dictionary[entry] / max_value)
        if value <= upperbound and value >= lowerbound:
            normalized_dict[entry] = value

    return normalized_dict

#ranks sentence based of normalized word frequencies and proper noun content
def rank_sentences(word_frequencies, text):
    word_sentences = [word_tokenize(sents.lower()) for sents in text]
    ranking = {}
    for r in range(len(word_sentences)):
        ranking[r] = 0
    for r in range(len(word_sentences)):
        propernoun_list = proper_nouns(text[r])
        for word in word_sentences[r]:
            if word in word_frequencies.keys() and word_frequencies[word] >= .2:
                ranking[r] += word_frequencies[word]
            elif word in propernoun_list: #rank increased for each NNP the sentence contains
                ranking[r] += .2
    return ranking

#determines part of speech for each word in a sentence
#returns NNP that are in a given sentence
def proper_nouns(sentence):
    tagged_sent = pos_tag(sentence.split())
    propernouns = [word.lower() for word,pos in tagged_sent if pos == 'NNP']
    return set(propernouns)

#determines the similarity of two sentences based off common words
#returns a ratio of shared words to total words in two sentences
def sentence_similarity(senta, sentb):
    words_a = word_tokenize(senta)
    words_b = word_tokenize(sentb)
    combo = set(words_a).union(set(words_b))
    return len(combo) / (len(words_a) + len(words_b))

#prints out the highest ranking sentences in order, excluding sentences that are too similar
#returns indexes of setences chosen for summary
def print_summary(text, ordered_rankings):
    summary_indexes = []
    num_sentences = len(text)
    summary_length = round(num_sentences * .25) #the decimal value determines the length of the summary
    sentence_indices = []
    for r in range(1, summary_length + 1):
        sentence_indices.append(ordered_rankings[-r][0])
    sentence_indices = sorted(sentence_indices)
    for s in range(len(sentence_indices)):
        if sentence_indices[s] == 0:
            print(text[sentence_indices[s]])
            summary_indexes.append(sentence_indices[s])
        elif sentence_similarity(text[s], text[s - 1]) > .7: #determines if the next sentence in the summary is too similar to the previous sentence
            print(text[sentence_indices[s]])
            summary_indexes.append(sentence_indices[s])
    return summary_indexes

#input filename of article to be summarized
#prints summary of article and returns proper noun coverage
def main(filename):
    f = open(filename, 'r', encoding="utf8")
    raw = f.read()
    stopword_list = stopwords.words('english')
    sents = sent_tokenize(raw)
    frequency_results = word_frequency(sents, stopword_list)
    rankings = rank_sentences(frequency_results, sents)
    ordered_rankings = sorted(rankings.items(), key=lambda ranking:ranking[1])
    results = print_summary(sents, ordered_rankings)
    proper_n = proper_nouns(raw)
    summary_words = []
    for entry in results:
        words = word_tokenize(sents[entry])
        for result in words:
            summary_words.append(result)
    counter = 0
    for entry in proper_n:
        if entry not in summary_words:
            counter += 1
    p_noun_coverage = (counter / len(proper_n))*100 #percentage of article's total proper nouns used in summary
    return p_noun_coverage
