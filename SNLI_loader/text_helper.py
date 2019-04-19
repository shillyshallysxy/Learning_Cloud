# Text Helper Functions
# ---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import collections
from tqdm import tqdm
import numpy as np


# Normalize text
def normalize_text(text):
    # Lower case
    text = text.lower()
    # Remove punctuation
    text = ''.join(c for c in text if c not in string.punctuation)
    # Trim extra whitespace
    text = ' '.join(text.split())
    return text


# Build dictionary of words
def build_dictionary(sentences, language='en', vocabulary_size=60000):
    # Turn sentences (list of strings) into lists of words
    if language == 'en':
        split_sentences = [s.split() for s in sentences]
    elif language == 'cn':
        split_sentences = [s for s in sentences]
    else:
        split_sentences = []
    words = [x for sublist in split_sentences for x in sublist]
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['_UNK', -1]]
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common())
    # Now create the dictionary
    word_dict = {'_PAD': 0, '_BEGIN': 1, '_EOS': 2}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in tqdm(count):
        word_dict[word] = len(word_dict)
        if len(word_dict) >= vocabulary_size:
            break
    return word_dict


# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict, ensure=None, language='en'):
    # Initialize the returned data
    data = []
    for i, sentence in tqdm(enumerate(sentences)):
        sentence_data = []
        # For each word, either use selected index or rare word index
        if language == 'en':
            split_sentences = sentence.split(' ')
        elif language == 'cn':
            split_sentences = sentence
        else:
            split_sentences = []
        for word in split_sentences:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = word_dict['_UNK']

            sentence_data.append(word_ix)

        if ensure is None:
            data.append(sentence_data)
        else:
            if ensure[i] == len(sentence_data):
                data.append(sentence_data)
            else:
                print(ensure[i], " doesn't match ", sentence_data)
    return data


# Turn text data into lists of integers from dictionary
def numbers_to_text(sentences, word_dict):
    word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        split_sentences = sentence
        for word in split_sentences:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = '_UNK'
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return data


def load_data(folder_name='G:\python\DeepLearning\Learning_Cloud\\temp\cn_nlp\corpus\\translate\Bilingual'):
    dirs = []
    en_data = []
    en_len = []
    cn_data = []
    cn_len = []
    temp = {}
    max_len = 50
    for dir_name in os.listdir(folder_name):
        for file_name in os.listdir(os.path.join(folder_name, dir_name)):
            dirs.append(os.path.join(folder_name, dir_name, file_name))
    for folder_name in tqdm(dirs):
        with open(folder_name, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                str_ = line.encode('utf-8', errors='ignore').decode().rstrip()
                if index % 2 == 0:
                    temp[0] = normalize_text(str_)
                    temp[1] = len(temp[0].split(' '))
                else:
                    temp[2] = str_
                    temp[3] = len(str_)
                    if temp[1] < max_len and temp[3] < max_len:
                        en_data.append(temp[0])
                        en_len.append(temp[1])
                        cn_data.append(temp[2])
                        cn_len.append(temp[3])
        f.close()
    return en_data, en_len, cn_data, cn_len


def load_test_data(folder_name=
                   'G:\python\DeepLearning\Learning_Cloud\\temp\cn_nlp\corpus\\translate\Testing\Testing-Data.txt'):
    en_data = []
    en_len = []
    cn_data = []
    cn_len = []
    temp = {}
    max_len = 50
    with open(folder_name, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            str_ = line.encode('utf-8', errors='ignore').decode().rstrip()
            if index % 2 == 0:
                temp[0] = normalize_text(str_)
                temp[1] = len(temp[0].split(' '))
            else:
                temp[2] = str_
                temp[3] = len(str_)
                if temp[1] < max_len and temp[3] < max_len:
                    en_data.append(temp[0])
                    en_len.append(temp[1])
                    cn_data.append(temp[2])
                    cn_len.append(temp[3])
        f.close()
    return en_data, en_len, cn_data, cn_len
