# Text Helper Functions
# ---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import tarfile
import collections
import numpy as np
import requests
import codecs


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

    return (texts)


# Build dictionary of words
def build_dictionary(sentences, vocabulary_size=0, list=False):
    # Turn sentences (list of strings) into lists of words
    if not list:
        sentences = [s.split() for s in sentences]
    words = [x for sublist in sentences for x in sublist]

    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]
    if vocabulary_size == 0:
        count.extend(collections.Counter(words).most_common())
    else:
        # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)

    return (word_dict)


# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict, list=False):
    # Initialize the returned data
    data = []
    if not list:
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
    else:
        for sentence in sentences:
            sentence_data = []
            # For each word, either use selected index or rare word index
            for word in sentence:
                if word in word_dict:
                    word_ix = word_dict[word]
                else:
                    word_ix = 0
                sentence_data.append(word_ix)
            data.append(sentence_data)
    return (data)


# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                            enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        # Pull out center word of interest for each window and create a tuple for each window
        batch, labels = [], []
        if method == 'skip_gram':
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
            batch_and_labels = [(rand_sentence[ix:ix+window_size], rand_sentence[ix+window_size])
                            for ix, x in enumerate(rand_sentence) if ix < len(rand_sentence) - window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            batch = [x+[rand_sentence_ix] for x in batch]
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


# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    # Check if files are already downloaded
    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open('temp_movie_review_temp.tar.gz', 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # Extract tar.gz file into temp folder
        tar = tarfile.open('temp_movie_review_temp.tar.gz', "r:gz")
        tar.extractall(path='temp')
        tar.close()

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding='latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]

    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data)

    return (texts, target)


def load_data(path, path2=''):
    data_folder_name = '..\\temp'
    data_path_name = 'cn_nlp'
    lines = []
    labels = []
    lens = []
    input_data = codecs.open(os.path.join(data_folder_name, data_path_name, path), 'r', 'utf-8')
    for ln in input_data.readlines():
        words_labels = ln.strip().split()
        line = []
        label = []
        for wl in words_labels:
            word_label = wl.split('/')
            line.append(word_label[0])
            tag = word_label[-1]
            tag_num = 0
            if tag == 'S':
                tag_num = 0
            elif tag == 'B':
                tag_num = 1
            elif tag == 'M':
                tag_num = 2
            elif tag == 'E':
                tag_num = 3
            label.append(tag_num)
        if 200 > len(label) > 1:
            lines.append(line)
            labels.append(label)
            lens.append(len(label))
    input_data.close()
    if not path2 == '':
        input_data2 = codecs.open(os.path.join(data_folder_name, data_path_name, path2), 'r', 'utf-8')
        for ln in input_data2.readlines():
            words_labels = ln.strip().split()
            line = []
            label = []
            for wl in words_labels:
                word_label = wl.split('/')
                line.append(word_label[0])
                tag = word_label[-1]
                tag_num = 0
                if tag == 'S':
                    tag_num = 0
                elif tag == 'B':
                    tag_num = 1
                elif tag == 'M':
                    tag_num = 2
                elif tag == 'E':
                    tag_num = 3
                label.append(tag_num)
            if 200 > len(label) > 1:
                lines.append(line)
                labels.append(label)
                lens.append(len(label))
        input_data2.close()
    return lines, labels, np.array(lens)
