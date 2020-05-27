# Code adapted from original code by Robert Guthrie

import os, sys, optparse, gzip, re, logging, string, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

def prepare_sequence(seq, to_ix, unk):
    idxs = []
    if unk not in to_ix:
        idxs = [to_ix[w] for w in seq]
    else:
        idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
    return torch.tensor(idxs, dtype=torch.long)

def create_char_level_vectors(sentence, word_to_ix, char_to_ix):
    width = len(char_to_ix)
    vectors = torch.tensor(0)
    first = True
    for word in sentence:
        v1 = torch.zeros([width])
        v2 = torch.zeros([width])
        v3 = torch.zeros([width])
        "1. Create a one-hot vector v1 for the first character of the word."
        v1[char_to_ix[word[0]]] = 1

        "2. Create a vector v2 where the index of a character has the count of that character in the word."
        for c in word[1:-1]:
            v2[char_to_ix[c]] = v2[char_to_ix[c]] + 1

        "3. Create a one-hot vector v3 for the last character of the word."
        v3[char_to_ix[word[-1]]] = 1

        "4. Concatenate three vectors to get final character level representation."
        v = torch.cat((v1,v2,v3),dim=0).view(1,300)
        if first:
            vectors = v
            first = False
        else:
            vectors = torch.cat((vectors,v),dim=0)
    return vectors

# def generate_noise(training_data):
#     random_seed = 12345
#     new_training_data = []
#     for sentence, tags in training_data:
#         new_sentence = []
#         for word in sentence:
#             if(len(word) >= 3):
#                 random.seed(random_seed) 
#                 # Adding seed making sure get the same result for next time with same training dataset
#                 rand = random.randint(0, 2)
#                 random_seed += 1

#                 random.seed(random_seed)
#                 target_index = random.randint(1, len(word)-2)
#                 random_seed += 1

#                 # Swap
#                 if rand == 2:
#                     new_word = word[:target_index] + word[target_index + 1] + word[target_index] 
#                 # Drop  
#                 if rand == 1:
#                     new_word = word[:target_index] + word[target_index+1:]
#                 # Normal
#                 if rand == 0:
#                     new_word = word
#             else:
#                 new_word = word
#             new_sentence.append(new_word)
#         # adding noise to the word in sentence
#         new_training_data.append((new_sentence, tags))
#     #random.shuffle(new_training_data)
#     return new_training_data

class LSTMTaggerModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        torch.manual_seed(1)
        super(LSTMTaggerModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim+300, hidden_dim, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, char_vectors):
        embeds = self.word_embeddings(sentence)

        "Concatenate the sentence length 2D tensor of character vectors with the word embeddings"
        embeds = torch.cat((embeds,char_vectors), dim=1)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class LSTMTagger:

    def __init__(self, trainfile, modelfile, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64):
        self.unk = unk
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.training_data = []
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        self.word_to_ix = {} # replaces words with an index (one-hot vector)
        self.char_to_ix = {} # replaces character with an index
        self.tag_to_ix = {} # replace output labels / tags with an index
        self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag

        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        for c in string.printable:
            self.char_to_ix[c] = string.printable.find(c)

        logging.info("char_to_ix:", self.char_to_ix)
        logging.info("word_to_ix:", self.word_to_ix)
        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)

        self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def argmax(self, seq):
        output = []
        with torch.no_grad():
            inputs = prepare_sequence(seq, self.word_to_ix, self.unk)
            char_vectors = create_char_level_vectors(seq, self.word_to_ix, self.char_to_ix)
            tag_scores = self.model(inputs,char_vectors)
            for i in range(len(inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output

    def train(self):
        loss_function = nn.NLLLoss()

        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            # random.shuffle(generate_noise(self.training_data))
            for sentence, tags in tqdm.tqdm(self.training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
                targets = prepare_sequence(tags, self.tag_to_ix, self.unk)

                # Step 2.1. Create character level vector representation of size n x 300
                char_vectors = create_char_level_vectors(sentence, self.word_to_ix, self.char_to_ix)

                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in, char_vectors)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets) * 2
                loss.backward()
                self.optimizer.step()

            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        loss = saved_model['loss']
        self.unk = saved_model['unk']
        self.word_to_ix = saved_model['word_to_ix']
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        self.model.eval()
        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax(sent))
        return decoder_output

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
    optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'chunker'), help="filename without suffix for model files")
    optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
    optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
    optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    if opts.modelfile[-4:] == '.tar':
        modelfile = opts.modelfile[:-4]
    chunker = LSTMTagger(opts.trainfile, modelfile, opts.modelsuffix, opts.unk)
    # use the model file if available and opts.force is False
    if os.path.isfile(opts.modelfile + opts.modelsuffix) and not opts.force:
        decoder_output = chunker.decode(opts.inputfile)
    else:
        print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
        chunker.train()
        decoder_output = chunker.decode(opts.inputfile)

    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
