###############################################################################
#######################  1. LOAD THE TRAINING TEXT  ###########################
###############################################################################
with open("data/text8.txt") as f:
    text = f.read()
    
  
###############################################################################
###########################  2. PRE-PROCESS TEXT  #############################
###############################################################################
import utils
words = utils.preprocess(text)


###############################################################################
#########################  3. CREATE DICTIONARIES  ############################
###############################################################################
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]


###############################################################################
#########################  4. PERFORM SUBSAMPLING   ###########################
###############################################################################
from collections import Counter
import numpy as np
import random 

freq = Counter(int_words)
threshold = 1e-5
# probability that word i will be dropped while subsampling = p_drop[i]
p_drop = {word: 1 - np.sqrt(threshold / (freq[word])/len(int_words)) for word in freq}
train_words = [word for word in int_words if p_drop[word] < (1 - random.random())]


###############################################################################
################  5. GET CONTEXT TARGETS FOR EACH WORD INPUT   ################
###############################################################################
# get a random range of 1-5 context word targets on the left & right of the word input with index 'idx' from 'words'
def get_targets(words, idx, window_size=5):
    
    R = random.randint(1, 5)
    start = max(0,idx-R)
    end = min(idx+R,len(words)-1)
    targets = words[start:idx] + words[idx+1:end+1] # +1 since doesn't include this idx
    
    return targets


###############################################################################
######################  6. GET (INPUT, OUTPUT) BATCHES  #######################
###############################################################################
def get_batches(words, batch_size, window_size = 5):
    
    for i in range(0, len(words), batch_size):
        curr = words[i:i + batch_size]   # current batch
        batch_x, batch_y = [], []
        
        for ii in range(len(curr)):
            x = [curr[ii]]
            y = get_targets(curr, ii)
            batch_x.extend(x * len(y))
            batch_y.extend(y)
        
        yield batch_x, batch_y
        

###############################################################################
##############  7. DEFINE VALIDATION METRIC: COSINE SIMILARITY  ###############
###############################################################################
import torch

def cosine_similarity(embedding, n_valid_words=16, valid_window=100):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        embedding: PyTorch embedding module
        n_valid_words: # of validation words (recommended to have even numbers)
    """
    all_embeddings = embedding.weight  # (n_vocab, n_embed) 
    # sim = (a . b) / |a||b|
    magnitudes = all_embeddings.pow(2).sum(dim=1).sqrt().unsqueeze(0) # (1, n_vocab)
  
    # Pick validation words from 2 ranges: (0, window): common words & (1000, 1000+window): uncommon words 
    valid_words = random.sample(range(valid_window), n_valid_words//2) + random.sample(range(1000, 1000+valid_window), n_valid_words//2)
    valid_words = torch.LongTensor(np.array(valid_words)).to(device) # (n_valid_words, 1)

    valid_embeddings = embedding(valid_words) # (n_valid_words, n_embed)
    # (n_valid_words, n_embed) * (n_embed, n_vocab) --> (n_valid_words, n_vocab) / 1, n_vocab)
    similarities = torch.mm(valid_embeddings, all_embeddings.t()) / magnitudes  # (n_valid_words, n_vocab)
  
    return valid_words, similarities
        

###############################################################################
################  8. DEFINE SKIP GRAMS w/ NEG SAMPLING MODEL  #################
###############################################################################
from torch import nn
from torch.functional import F

class SkipGram_NegSample_Model(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors  # input vector embeddings
    

    def forward_target(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors  # output vector embeddings
    

    def forward_noise(self, batch_size, n_samples=5):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        # If no Noise Distribution specified, sample noise words uniformly from vocabulary
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # torch.multinomial :
        # Returns a tensor where each row contains (num_samples) **indices** sampled from 
        # multinomial probability distribution located in the corresponding row of tensor input.
        noise_words = torch.multinomial(input       = noise_dist,           # input tensor containing probabilities
                                        num_samples = batch_size*n_samples, # number of samples to draw
                                        replacement = True)
        noise_words = noise_words.to(device)
        
        # use context matrix for embedding noise samples
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors


###############################################################################
#########################  9. DEFINE LOSS FUNCTION  ###########################
###############################################################################
class SkipGram_NegSample_Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, 
              input_vectors, 
              output_vectors, 
              noise_vectors):
      
    batch_size, embed_size = input_vectors.shape
    
    input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
    output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors
    
    # correct log-sigmoid loss
    out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()
    
    # incorrect log-sigmoid loss
    noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
    noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

    return -(out_loss + noise_loss).mean()  # average batch loss


###############################################################################
######################  10. DEFINE NOISE DISTRIBUTION  ########################
###############################################################################
# As defined in the paper by Mikolov et all.
freq_ratio = {word: count/len(vocab_to_int) for word, count in freq.items()}        
freq_ratio = np.array(sorted(freq_ratio.values(), reverse = True))
unigram_dist = freq_ratio / freq_ratio.sum() 
noise_dist = torch.from_numpy(unigram_dist**0.75 / np.sum(unigram_dist**0.75))


###############################################################################
##################  11. DEFINE MODEL, LOSS, & OPTIMIZER  ######################
###############################################################################
from torch import optim

embedding_dim = 300
model = SkipGram_NegSample_Model( len(vocab_to_int), embedding_dim, noise_dist )
criterion = SkipGram_NegSample_Loss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


###############################################################################
##########################  12. TRAIN THE NETWORK!  ###########################
###############################################################################
print_every = 1500
step = 0
n_epochs = 5
device = 'cuda' if torch.cuda.is_available else 'cpu'

def train_skipgram(model,
                   criterion,
                   optimizer,
                   int_words,
                   n_negative_samples=5,
                   batch_size=512,
                   n_epochs=5,
                   print_every=1500,
                   ):
    model.to(device)
    
    step = 0
    for epoch in range(n_epochs):
        for inputs, targets in get_batches(int_words, batch_size=batch_size):
            step += 1
            inputs = torch.LongTensor(inputs).to(device)    # [b*n_context_words]
            targets = torch.LongTensor(targets).to(device)  # [b*n_context_words]
            
            embedded_input_words = model.forward_input(inputs)
            embedded_target_words = model.forward_target(targets)
            embedded_noise_words = model.forward_noise(batch_size=inputs.shape[0], 
                                                      n_samples=n_negative_samples)

            loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step % print_every) == 0:
              print("Epoch: {}/{}".format((epoch+1), n_epochs))
              print("Loss: {:.4f}".format(loss.item()))
              valid_idxs, similarities = cosine_similarity(model.in_embed)
              _, closest_idxs = similarities.topk(6)
              valid_idxs, closest_idxs = valid_idxs.to('cpu'), closest_idxs.to('cpu')
              
              for ii, v_idx in enumerate(valid_idxs):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[v_idx.item()] + " | "+ ", ".join(closest_words))

              print("...")
            
 train_skipgram(model,
               criterion,
               optimizer,
               int_words,
               n_negative_samples=5)


###############################################################################
##############  13. VISUALIZE EMBEDDED WORD VECTORS USING TSNE  ###############
###############################################################################
from sklearn import TSNE
import matplotlib.pyplot as plt

embeddings = model.embedding.weight.to('cpu').data.numpy()
n_viz_words = 400  # plot only first 400 words from vocabulary
tsne = TSNE()
embeddings_tsne = TSNE.fit_transform(embeddings[:n_viz_words, :])

fig, ax = plt.subplots(figsize = (16, 16))
for i in range(n_viz_words):
    plt.scatter(*embeddings_tsne[i, :], color = 'steelblue')
    plt.annotate(int_to_vocab[i], (embeddings_tsne[i, 0], embeddings_tsne[i, 1]), alpha = 0.7)
