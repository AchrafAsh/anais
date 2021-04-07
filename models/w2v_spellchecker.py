import re
from collections import Counter
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

def model(filename):
    model = KeyedVectors.load_word2vec_format(
        filename,
        binary=True)
    
    words = model.index2word
    w_rank = {}
    for i,word in tqdm(enumerate(words)):
        w_rank[word] = i

    vocab = w_rank

    return model, vocab

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - vocab.get(word, 0)

def correction(vocab, word): 
    "Most probable spelling correction for word."
    return max(candidates(vocab, word), key=P)

def candidates(vocab, word): 
    '''Generate possible spelling corrections for word.
    '''
    return (known([word], vocab) 
            or known(vocab, edits1(word)) 
            or known(vocab, edits2(word)) 
            or [word])

def known(vocab, words): 
    '''The subset of `words` that appear in the dictionary of vocab.'''
    return set(w for w in words if w in vocab)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

if __name__ == "__main__":
    model, vocab = model("./GoogleNews-vectors-negative300.bin.gz")
    correction(vocab, "petersbourg")