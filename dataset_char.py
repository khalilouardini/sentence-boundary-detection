import re
from collections import defaultdict
import numpy as np

# This code was taken from https://github.com/dbmdz/deep-eos/
def build_data_set_char(t, window_size=4):
  """ Builds data set from corpus
  This method builds a dataset from the training corpus
  # Arguments
    t          : Input text
    window_size: The window size for the current model
  # Returns
    A data set which contains char sequences as feature vectors
  """

  data_set_char_eos = \
      [(1.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
        t[m.start():m.start() + window_size + 1].replace("\n", " "))
        for m in re.finditer('[\.:?!;][^\n]?[\n]', t)]

  data_set_char_neos = \
      [(0.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
        t[m.start():m.start() + window_size + 1].replace("\n", " "))
        for m in re.finditer('[\.:?!;][^\s]?[ ]+', t)]

  return data_set_char_eos + data_set_char_neos

def build_char_2_id_dict(data_set_char, min_freq=10000):
    """ Builds a char_to_id dictionary
    This methods builds a frequency list of all chars in the data set.
    Then every char gets an own and unique index. Notice: the 0 is reserved
    for unknown chars later, so id labelling starts at 1.
    # Arguments
      data_set_char: The input data set (consisting of char sequences)
      min_freq     : Defines the minimum frequecy a char must appear in data set
    # Returns
      char_2_id dictionary
    """
    char_freq = defaultdict(int)
    char_2_id_table = {}

    for char in [char for label, seq in data_set_char for char in seq]:
        char_freq[char] += 1

    id_counter = 1

    for k, v in [(k, v) for k, v in char_freq.items() if v >= min_freq]:
        char_2_id_table[k] = id_counter
        id_counter += 1

    return char_2_id_table

def build_data_set(data_set_char, char_2_id_dict, window_size):
    """ Builds a "real" data set with numpy compatible feature vectors
    This method converts the data_set_char to real numpy compatible feature
    vectors. It does also length checks of incoming and outgoing feature
    vectors to make sure that the exact window size is kept
    # Arguments
      data_set_char : The input data set (consisting of char sequences)
      char_2_id_dict: The char_to_id dictionary
      window_size   : The window size for the current model
    # Returns
      A data set which contains numpy compatible feature vectors
    """

    data_set = []

    for label, char_sequence in data_set_char:
        ids = []

        if len(char_sequence) == 2 * window_size + 1:
            for char in char_sequence:
                if char in char_2_id_dict:
                    ids.append(char_2_id_dict[char])
                else:
                    ids.append(0)

            feature_vector = np.array([float(ids[i])
                                        for i in range(0, len(ids))], dtype=float)

            data_set.append((float(label), feature_vector))

    return data_set

def prepare_data(input_file):
    with open(input_file, mode='r', encoding='utf-8') as f:
        training_corpus = f.read()
        
    data_set_char = build_data_set_char(training_corpus, window_size=4)
    char_2_id_dict = build_char_2_id_dict(data_set_char, min_freq=10000)
    data_set = build_data_set(data_set_char, char_2_id_dict,  window_size=4)
                                      
    X = np.array([i[1] for i in data_set])
    y = np.array([i[0] for i in data_set])
    vocab_size = len(data_set_char)

    return X, y, vocab_size
    