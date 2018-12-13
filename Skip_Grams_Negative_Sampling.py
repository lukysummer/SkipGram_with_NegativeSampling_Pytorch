###############################################################################
#######################  1. LOAD THE TRAINING TEXT  ###########################
###############################################################################
with open("data/text8.txt") as f:
    text = f.read()
    
  
###############################################################################
##########################  2. TEXT PRE-PROCESSING  ###########################
###############################################################################
import utils
words = utils.preprocess(text)


###############################################################################
#########################  3. CREATE DICTIONARIES  ############################
###############################################################################
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]


###############################################################################
#############################  4. SUBSAMPLING   ###############################
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
# get a random range of 1-5 context word targets for word input with index 'idx' from 'words'
def get_targets(words, idx, window_size = 5):
    
    R = random.randint(1, 5)
    targets = words[max(0, idx-R) : idx] + words[idx+1 : min(idx+R+1, len(words))]
    
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
#################  7. VALIDATION METRIC: COSINE SIMILARITY  ###################
###############################################################################
import torch

def cosine_similarity(embedding, valid_size = 16, valid_window = 100, device = 'cpu'):
    embed_vectors = embedding.weight  # shape: (n_vocab, embed_dim) 
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    valid_idxs = random.sample(range(valid_window), valid_size/2)
    valid_idxs = np.append(valid_idxs, 
                           random.sample(range(1000, 1000+valid_window), valid_size/2))
    valid_idxs = torch.LongTensor(valid_idxs).to(device) # (16, 1)
    valid_vectors = embedding(valid_idxs) # (16, embed_dim)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes  # (16, n_vocab)
    
    return valid_idxs, similarities
        

###############################################################################
################  8. DEFINE SKIP GRAMS w/ NEG SAMPLING MODEL  #################
###############################################################################
from torch import nn
from torch.functional import F

class SkipGram_NegSample_Model(nn.module):
    
    def __init__(self, n_vocab, n_embed, noise_dist = None):
        super().__init__()
        self.n_vcab = n_vocab  # number of UNIQUE words in vocabulary
        self.n_embed = n_embed  # embedding dimension
        self.noise_dist = noise_dist  # probability distribution to pick noise samples from vocabulary
        
        self.input_embed = nn.Embedding(n_vocab, n_embed)
        self.output_embed = nn.Embeddimg(n_vocab, n_embed)
        
        # initialize weights (= embedding tables) --> may help converge faster
        self.input_embed.weight.uniform_(-1, 1)
        self.output_embed.weight.uniform(-1, 1)
        
        
    def forward_input (self, input_words):
        embedded_input_words = self.input_embed(input_words)
        
        return embedded_input_words
    
    
    def forward_target (self, target_words):
        embedded_target_words = self.output_embed(target_words)
        
        return embedded_target_words
    
    
    def forward_noise (self, batch_size, n_samples):
        # If no Noise Distribution specified, sample noise words uniformly from vocabulary
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)  # shape : [n_vocab]
        else:
            noise_dist = self.noise_dist   # shape : [n_vocab]
            
        noise_words = torch.multinomial(noise_dist, batch_size * n_samples, replacement = True)
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        noise_words = noise_words.to(device)
        
        embedded_noise_words = self.output_embed(noise_words).view(batch_size,
                                                                   n_samples,
                                                                   self.n_embed)
        
        return embedded_noise_words


###############################################################################
#########################  9. DEFINE LOSS FUNCTION  ###########################
###############################################################################
class SkipGram_NegSample_Loss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    
    def forward (self, embedded_input_words, embedded_target_words, embedded_noise_words):
        
        n_batches, n_embed = embedded_input_words.shape
        
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        embedded_input_words = torch.LongTensor(embedded_input_words).to(device)
        # make embedded_input_words a batch of Column Vectors
        embedded_input_words = embedded_input_words.view(n_batches, n_embed, 1)  
        
        embedded_target_words = torch.LongTensor(embedded_target_words).to(device)
        # make embedded_target_words a batch of Row Vectors
        embedded_target_words = embedded_target_words.view(n_batches, 1, n_embed)  
        
        ## Define Loss Components
        target_loss = torch.bmm(embedded_target_words, embedded_input_words).sigmoid().log()
        target_loss = target_loss.squeeze()
        
        noise_loss = torch.bmm(embedded_noise_words.neg(), embedded_input_words).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(dim=1)
    
        return -(target_loss + noise_loss).mean()  # return average batch loss


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
optimizer = optim.Adam(model.parameters(), lr = 0.003)


###############################################################################
##########################  12. TRAIN THE NETWORK!  ###########################
###############################################################################
print_every = 1500
step = 0
n_epochs = 5
device = 'cuda' if torch.cuda.is_available else 'cpu'

for epoch in range(n_epochs):
    for inputs, targets in get_batches(int_words, batch_size = 512):
        step += 1
        inputs, targets = torch.LongTensor(inputs).to(device), torch.LongTensor(targets).to(device)
        
        embedded_input_words = model.forward_input(inputs)
        embedded_target_words = model.forward_target(targets)
        embedded_noise_words = model.forward_noise(inputs.shape[0], n_samples = 5)

        loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step % print_every) == 0:
            print("Epoch: {}/{}".format((epoch+1), n_epochs))
            print("Loss: {:.4f}".format(loss.item()))
            valid_idxs, similarities = cosine_similarity(model.embed, device)
            _, closest_idxs = similarities.topk(6)
            valid_idxs, closest_idxs = valid_idxs.to(device), closest_idxs.to(device)
            
            for ii, v_idx in enumerate(valid_idxs):
                closest_words = [int_to_vocab[idx] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[v_idx] + " | "+ ", ".join(closest_words))
                
            print("...")


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