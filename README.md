# Skip Grams with Negative Sampling in PyTorch

This is my implementation of **Skip Grams**, which is trained to:

1. Generate vector representations of words in vocabulary using **Word2Vec Embeddings**
2. Using the embedded vectors, generate semantically similar words to a given input word,
 by looking at the cosine distance between the input and all other words in the vocabulary
   
   
Using **Negative Sampling**, the model tries to minimize the following **Loss Function**:


<img src = "loss_function.png">


## Repository 

This repository contains:
* **Skip_Grams_Negative_Sampling.py** : Complete code for implementing facie generation task using DCGAN
			
			
## List of Hyperparameters used:

* Batch Size = **512**
* Threshold for Subsampling = **1e-5**  
* Window Size for Context = **5**  
* Embedding Dimension = **300**
* Number of Negative (Noise) Samples = **5**
* Learning Rate = **0.003**
* Number of Epochs = **5**


## Sources

I referenced the following sources for building & debugging the final model :

* https://github.com/udacity/deep-learning-v2-pytorch



