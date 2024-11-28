import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# Hyperparameters
EMBEDDING_DIM = 50
WINDOW_SIZE = 3
NEGATIVE_SAMPLES = 3
BATCH_SIZE = 32
EPOCHS = 5

# Preprocessing
def preprocess(data):
    tokenized = data.split()
    vocab = Counter(tokenized)
    word2index = {word: idx for idx, word in enumerate(vocab.keys())}
    index2word = {idx: word for word, idx in word2index.items()}
    unigram_counts = np.array([vocab[word] for word in vocab])
    unigram_probs = (unigram_counts ** 0.75) / np.sum(unigram_counts ** 0.75)
    return tokenized, word2index, index2word, unigram_probs

def generate_skipgrams(tokenized, word2index, window_size):
    skipgrams = []
    for idx, center_word in enumerate(tokenized):
        center = word2index[center_word]
        start = max(0, idx - window_size)
        end = min(len(tokenized), idx + window_size + 1)
        for i in range(start, end):
            if i != idx:
                context = word2index[tokenized[i]]
                skipgrams.append((center, context))
    return skipgrams

def negative_sampling(batch_size, unigram_probs, vocab_size, k):
    neg_samples = np.random.choice(vocab_size, size=(batch_size, k), p=unigram_probs)
    return torch.tensor(neg_samples, dtype=torch.long)

# Word2Vec Model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.input_embeddings.weight.data, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.zeros_(self.output_embeddings.weight.data)

    def forward(self, center_words, outside_words, negative_samples):
        center_embedding = self.input_embeddings(center_words)
        outside_embedding = self.output_embeddings(outside_words)
        neg_embedding = self.output_embeddings(negative_samples)

        pos_score = torch.sum(center_embedding * outside_embedding, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score)).mean()

        neg_score = torch.bmm(neg_embedding, center_embedding.unsqueeze(2)).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_score)).mean()

        return -(pos_loss + neg_loss)

# Main Train Function
def train(data: str):
    tokenized, word2index, index2word, unigram_probs = preprocess(data)
    skipgrams = generate_skipgrams(tokenized, word2index, WINDOW_SIZE)
    vocab_size = len(word2index)

    model = Word2Vec(vocab_size, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    def create_batches(skipgrams, batch_size):
        random.shuffle(skipgrams)
        for i in range(0, len(skipgrams), batch_size):
            batch = skipgrams[i:i + batch_size]
            centers = torch.tensor([pair[0] for pair in batch], dtype=torch.long)
            contexts = torch.tensor([pair[1] for pair in batch], dtype=torch.long)
            yield centers, contexts

    for epoch in range(EPOCHS):
        for centers, contexts in create_batches(skipgrams, BATCH_SIZE):
            negative_samples = negative_sampling(len(centers), unigram_probs, vocab_size, NEGATIVE_SAMPLES)
            loss = model(centers, contexts, negative_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    embeddings = model.input_embeddings.weight.detach().numpy()
    w2v_dict = {index2word[idx]: embeddings[idx] for idx in range(vocab_size)}
    return w2v_dict
