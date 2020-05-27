import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
from heapq import heappush, heappop

maximum_word_length = 6
lam = 0.7125
# test.out score: 0.77
# dev.out score: 0.92

# maximum_word_length = 8
# lam = 0.5
# test.out score: 0.75
# dev.out score: 0.91

# maximum_word_length = 6
# lam = 0.5
# test.out score: 0.77
# dev.out score: 0.92

# maximum_word_length = 6
# lam = 0.98
# dev.out score: 0.90
# test.out score: 0.71

# maximum_word_length = 6
# lam = 0.9
# test.out score: 0.77
# dev.out score: 0.91

# maximum_word_length = 6
# lam = 0.7
# dev.out score: 0.92
# test.out score: 0.77



class Entry:
# each entry in the chart has four components: Entry(word, start-position, log-probability, back-pointer)
# the back-pointer in each entry links it to a previous entry that it extends
    def __init__(self, word, start_position, log_probability, back_pointer=None):
        self.word = word
        self.start_position = start_position # start position
        self.log_probability = log_probability # log probability
        self.back_pointer = back_pointer # back_pointer
    def __lt__(self, other):
        return  self.log_probability < other.log_probability
    def __eq__(self, other):
        # return (self.word, self.start_position, self.log_probability, self.backptr) == (other.word, other.start_position, other.log_probability, other.back_pointer)

        return (self.word, self.start_position, self.log_probability, self.back_pointer) == (other.word, other.start_position, other.log_probability, other.back_pointer)
    def __str__(self):
        return 'Entry(word='+self.word+', logprob='+str(self.log_probability)+ ')'

" We will need the function for calculating conditional of Probability"
def cPw(word, prev):
    try:
       value = PBi[prev + ' ' + word]/float(PUi[prev])
       return value
    except KeyError:
       return PUi(word)

class Segment:

    def __init__(self, PUi, PBi):
        self.PUi = PUi
        self.PBi = PBi
    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []

        ## Initialize the heap ##  
        heap = [] # a list or priority queue containing the entries to be expanded, sorted on start-position or log-probability
        # for each word that matches input at position 0
        # Do not need to looks for character more than 11, as the max len in count_1w is 11
        for i in range(min(len(text), maximum_word_length)):
            word = text[:i+1]
            if word in PUi or len(word) <= 2: # Check if word exists fron counts1w or Pw
                heappush(heap, Entry(word, 0, log10((1-lam)*PUi(word)))) #  insert Entry (word, 0, logPw(word), None) to heap
        chart = [None] * len(text)
        finalindex = 0
        while heap:
            entry = heappop(heap) # top entry in the heap
            endindex = len(entry.word) -1  + entry.start_position
            if finalindex < endindex :
                finalindex = endindex
            if chart[endindex]:  # if chart[endindex] exists entry
                preventry = chart[endindex]
                if entry.log_probability > preventry.log_probability: # then we check the current if has higher log probability
                    chart[endindex] = entry
                else:
                    continue
            else:  # if not previous entry for chart[endindex], chart[endindex] not exists
                chart[endindex] = entry # store current entry to chart[endindex]
        # for each newword that matches input starting at position endindex+1
            for i in range (endindex+1, min(len(text), maximum_word_length +endindex+1)):
                newword = text[endindex+1:i+1]
                try:
                    cPw = PBi[entry.word + ' ' + newword]/float(PUi[entry.word])
                except KeyError:
                    cPw = PUi(word)
                newentry = Entry(newword, endindex+1, entry.log_probability + log10(lam*cPw+(1-lam)*PUi(newword)), entry)
                if not newentry in heap:
                    heappush(heap, newentry)

        # ## Get the best segmentation ##
        finalindex = len(text) - 1
        finalentry = chart[finalindex]
        segmentation = []

        while finalentry:
            segmentation.insert(0,finalentry.word)
            finalentry = finalentry.back_pointer
        return segmentation # return array of best segmentation, and " ".join used later to combine all array string items

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: 
            return self[key]/(self.N *len(key)/2)
        else: 
            return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def reduce_probability_for_long_word(word, N):   
    return 10./(N * 10 ** (len(word)*11))

def returnZero(word, N):   
    return 0

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    N = 1024908267229

    PUi = Pdist(data=datafile(opts.counts1w),N=N, missingfn=reduce_probability_for_long_word)
    PBi = Pdist(data=datafile(opts.counts2w),N=N, missingfn=returnZero)
    segmenter = Segment(PUi, PBi)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))

# You can also optionally use the data used to create the above count files which is in train.txt.bz2 in the data directory.
# You can also use a larger dataset provided to you via this link: wseg_simplified_cn.txt.bz2 which contains 1M Chinese sentences with word segments (bzip2 compressed). This link is only available for SFU students. You must not give a copy of this data set to anybody.


# using the Algorithm: Iterative segmenter psesudo code and implement the baseline model with unigram
# As mentioned in HW0, Implementing a greedy search gets an F-score of 0.66 on dev 
# while the Baseline method with unigram counts gets 0.89 on the dev set.

# Improvement:
# Use the bigram model to score word segmentation candidates.
# Do better smoothing of the unigram and bigram probability models.
# More advanced methods1


# First, we need to implement the baseline model with iterative appoarch to replace the recursive approach with memoization
# The baseline mode is used unigram language model over Chinese words