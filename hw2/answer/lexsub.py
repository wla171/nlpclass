import os, sys, optparse
import tqdm
import pymagnitude
import numpy as np
import math
from copy import deepcopy
from numpy import dot
from numpy.linalg import norm

# alpha = 1
# beta = 2.19
# dev.out score: 48.32

def retrofitting(wordVectors, lexicon, numberOfIteration, alpha=1, beta=2.19):
    # Q to be equal to Q_hat
    # Initialize Q to be equal to the vectors in q_hat
    q_hat = wordVectors
    q = deepcopy(wordVectors) # copy a mutable new q vector

    # find connected vocab between (wi, wj)
    wvecKey = set(wordVectors.keys())
    lexcionKey = set(lexicon.keys())
    
    # if (wi,wj) are connected by an edge, get word which append in lexicon and word embedding 
    
    exist_word_vector = wvecKey.intersection(lexcionKey)

    # Retrofitting, we want to get new Q which append in lexicon space, therefore, it will not update the

    # So if (wi,wj) are connected by an edge in the ontology then we want qi and qj to be close in vector space.
    # numberOfIteration = 10
    for iteration in range(numberOfIteration):
        # Do Update Q here
        # neighbors q 
        for word in exist_word_vector:
            # The defined neighbours word inside lexicon for this word  
            lexicon_words = set(lexicon[word])
            # Get the neighbours word exists in word vector
            neighbours_words = lexicon_words.intersection(wvecKey)

            num_neighbours_words = len(neighbours_words) 
            if(num_neighbours_words == 0): # No need to update if there is no neighours word found
                continue 

            # alpha = 1
            # beta = 1
            # sum number j of neighbours_words (xi*q_hat) 
            numerator = alpha * q_hat[word]
            # add sum number j of neighbours_words (beta *q) 

            for nei_word in neighbours_words:
                numerator += beta * q[nei_word] 

            denominator = num_neighbours_words * (alpha + beta)

            # update the word vector 
            q[word] = numerator / denominator

    return q # Return matrix of new vector Q here 


def load_wvecs(wvec_file):
    wordVectors = {}
    for key, vector in wvec_file:
        wordVectors[key] = np.zeros(len(vector), dtype=float)
        for index, vecVal in enumerate(vector):
            wordVectors[key][index] = float(vecVal)
        ''' normalize weight vector '''
        wordVectors[key] /= math.sqrt((wordVectors[key]**2).sum() + 1e-6)
    return wordVectors

def load_lexicon(lexicon_file):
    lexicon = {}
    for line in open(lexicon_file, 'r'):
        words = line.lower().strip().split()
        lexicon[words[0]] = [word for word in words[1:]]
    return lexicon

''' Write word vectors to file '''
def save_word_vecs(wordVectors, outFileName):
#   sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')  
  for word, values in wordVectors.items():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')      
  outFile.close()

def cos(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def pcos(a, b):
    return (cos(a,b) + 1) / 2

def incorporating_context_words(wvecs, wvecKey, sentence, index, substitutes, topn):
    # Incorporating Context Words
    sort_substitutes = []
    non_target_words_set = set()
    for word in sentence[:index]:
        non_target_words_set.add(word)
    for word in sentence[index+1:len(sentence)]:
        non_target_words_set.add(word)
    non_target_words = non_target_words_set.intersection(wvecKey)
    target = sentence[index]
    num_non_target_words = len(non_target_words)
    for substitute in substitutes:
        numerator = cos(wvecs[substitute], wvecs[target]) * num_non_target_words
        if(num_non_target_words == 0):
            continue
        for non_target_word in non_target_words:
            numerator = numerator + cos(wvecs[substitute], wvecs[non_target_word])
        denominator = num_non_target_words * 2 # divide by number of non target word, also divide by 2 for getting average after adding target and non target cos
        score = numerator/denominator
        sort_substitutes.append((substitute, score))
    sort_substitutes = sorted(sort_substitutes, key=lambda x: x[1], reverse=False)
    # print(sort_substitutes)
    return map(lambda x: x[0], sort_substitutes[:topn])

class LexSub:

    def __init__(self, retrofitted_magnitude, wvec_file, retrofitted_vector, topn=10):
        self.retrofitted_magnitude = pymagnitude.Magnitude(retrofitted_magnitude)  # This is the Q_hat vector  100 dimenstional GloVe word vectors
        self.topn = topn 
        self.wvecs = wvec_file
        self.wvecKey = set(self.wvecs.keys())
        self.retrofitted_vector = retrofitted_vector

    def substitutes(self, index, sentence, use_context_word=False):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        # return the 10 guess word after created the retrofitted word vectors
        substitutes = list(map(lambda k: k[0], self.retrofitted_magnitude.most_similar(sentence[index], topn=10)))
        # Incorporating Context Words
        if use_context_word:
            substitutes = incorporating_context_words(self.retrofitted_vector, self.wvecKey, sentence, index, substitutes, self.topn)
        return substitutes
        # return(list(map(lambda k: k[0], self.retrofitted_magnitude.most_similar(sentence[index], topn=self.topn))))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    # add path for lexicons file
    # wordnet-synonyms.txt
    optparser.add_option("-x", "--lexiconfile", dest="lexiconfile", default=os.path.join('data', 'lexicons','ppdb-xl.txt'), help="lexicon file")
    optparser.add_option("-r", "--rfwordvecfile", dest="rfwordvecfile", default=os.path.join('data', 'glove.6B.100d.retrofit.magnitude'), help="retrofitted word vectors file")
    optparser.add_option("-t", "--itr", dest="iteration", default=10, help="iteration")
    optparser.add_option("-a", "--alpha", dest="alpha", default=1, help="alpha")
    optparser.add_option("-b", "--beta", dest="beta", default=2.19, help="beta")

    # ppdb-xl
    # alpha = 1
    # beta = 1.0185
    # dev.out score: 44.9207


    # optparser.add_option("-r", action="context_word", dest="context_word", default=False)
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    retrain = False
    word_vector = load_wvecs(pymagnitude.Magnitude(opts.wordvecfile))
    new_retrofitted_magnitude = os.path.join('data', 'glove.6B.100d.retrofit.magnitude')
    if retrain:
        new_retrofitted_txt = os.path.join('data', 'glove.6B.100d.retrofit.txt')

        lexicon = load_lexicon(opts.lexiconfile)
        retrofitted_vector = retrofitting(word_vector, lexicon, opts.iteration, opts.alpha, opts.beta)
        # We need to do retrofitting here
        save_word_vecs(retrofitted_vector, new_retrofitted_txt) 
        os.system("python3 -m pymagnitude.converter -i " + new_retrofitted_txt + " -o " + new_retrofitted_magnitude)
    else:
        retrofitted_vector = load_wvecs(pymagnitude.Magnitude(opts.rfwordvecfile))

    lexsub = LexSub(new_retrofitted_magnitude, word_vector, retrofitted_vector, int(opts.topn))
    
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
