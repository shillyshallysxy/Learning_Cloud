# Text Helper Functions
#---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import collections
import numpy as np


# Normalize text
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    
    return(texts)


# Build dictionary of words
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]
    
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common())
    
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    
    return(word_dict, len(word_dict))
    

def build_frequency_table(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    word_count_total = 0
    for s in sentences:
        word_count_total += len(s.split())
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]

    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]

    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common())

    # Now create the dictionary
    freq_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        freq_dict[word] = word_count/word_count_total

    return (freq_dict)


# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)


def generate_batch_data(sentences, sentence_len, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences[0]), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        rand_len = sentence_len[rand_sentence_ix]
        # print('generate_batch_data')
        # Generate consecutive windows to look at
        window_sequences = [list(rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)])
                            for ix, x in enumerate(rand_sentence) if ix < rand_len]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        # Pull out center word of interest for each window and create a tuple for each window
        batch, labels = [], []
        if method == 'skipgram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
            if len(tuple_data) > 0:
                batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method == 'cbow':
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]

            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
            if len(batch_and_labels) > 0:
                batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method == 'doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i + window_size], rand_sentence[i + window_size])
                                for i in range(0, len(rand_sentence) - window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return batch_data, label_data


def load_movie_data(folder_name='G:\\python\DeepLearning\doc\entext8'):
    save_folder_name = folder_name
    file = os.path.join(save_folder_name, 'text8')
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    data = [x.rstrip() for x in data]
    texts = data
    target = []
    return texts, target
