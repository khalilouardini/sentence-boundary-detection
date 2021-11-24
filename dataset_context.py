from collections import defaultdict
import numpy as np
import re
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
from tqdm import tqdm, tqdm_notebook, tnrange


def preprocess_text(text, remove_stop_words=False, stemming=False):
    # 1. Remove punctuation from text
    #text = ''.join([c for c in text if c not in punctuation])
    
    # Remove digits
    text = re.sub('[0-9]+', ' ', text)
    
    # Remove space
    #text = re.sub(r'\s+', ' ', text)
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stopwords]
        # remove digits
        text = [w for w in text if not w.isdigit()]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stemming:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return text

def create_datset_context(text):
    # Pre-processing
    preprocess_text(text)
    # Positive samples
    dataset_words_pos = []
    markers = [t for t in re.finditer('[\.:?!;][^\n]?[\n]', text)]
    for m in markers:
        try:
          extract = text[m.start() - 100: m.start() + 100].replace("\n", " ")
          extract = extract.replace(". ", ".")
          match = m.group(0).replace("\n", "")
          before = ''.join([c.lower() for c in extract.split(match)[0] if c not in punctuation])
          before = [c.lower() for c in before.split(" ")[-3:] if c!='']
          after = ''.join([c.lower() for c in extract.split(match)[1] if c not in punctuation])
          after = [c.lower() for c in after.split(" ")[:3] if c!= '']
          context = before + after
          dataset_words_pos.append((1.0, context))
        except:
          continue
    # Negative samples
    dataset_words_neg = []
    markers = [t for t in re.finditer('[\.:?!;][^\s]?[ ]', text)]
    for m in markers:
        try:
          extract = text[m.start() - 100: m.start() + 100].replace("\n", " ")
          extract = extract.replace(". ", ".")
          match = m.group(0).replace("\n", "")
          match = match.replace(" ", "")
          before = ''.join([c for c in extract.split(match)[0] if c not in punctuation ])
          before = [c.lower() for c in before.split(" ")[-3:] if c!='']
          after = ''.join([c for c in extract.split(match)[1] if c not in punctuation])
          after = [c.lower() for c in after.split(" ")[:3] if c!= '']
          context = before + after
          dataset_words_neg.append((0.0, context))
        except:
          continue
    return dataset_words_pos + dataset_words_neg

def build_vocab(dataset_context):
    # build vocabulary and corresponding counts
    counts = Counter()
    for context in tqdm(dataset_context):
        counts.update(w.lower() for w in context[1])

    # sort with most frequently occuring words first
    size_vocab = len(counts)
    counts = {k: counts[k] / size_vocab for k in counts}
    #counts = {k: counts[k] for k in counts if counts[k] > 0.001}
    words = sorted(counts, key=counts.get, reverse=True)

    # add <pad> and <unk> token to vocab which will be used later
    words = ['_PAD','_UNK'] + words

    print("Size of vocabulary {}".format(len(words)))
    
    word2idx = {o:i for i,o in enumerate(words)}
    idx2word = {i:o for i,o in enumerate(words)}
    
    return word2idx, words

def indexer(context, word2idx): 
    return [word2idx[w.lower()] for w in context if w in word2idx]

def pad_data(s, max_len):
    padded = np.zeros((max_len,), dtype=np.int64)
    if len(s) > max_len:
        padded[:] = s[max_len]
    else:
        padded[:len(s)] = s
    return padded

def encode_features(text):
    text = preprocess_text(text)
    
    # create context dataset.
    dataset_context = create_datset_context(text)
    max_len = max([len(context[1]) for context in dataset_context])
    
    # Create vocab.
    word2idx, vocab = build_vocab(dataset_context)
    
    # Create features & pad data if necessary.
    X = np.array([pad_data(indexer(context[1], word2idx), max_len) for context in dataset_context])
    y = np.array([context[0] for context in dataset_context])
    
    return X, y, dataset_context, vocab, word2idx
