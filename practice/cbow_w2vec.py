import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# Hyperparameters
EMBEDDING_DIM = 100
WINDOW_SIZE = 2  # Количество слов с каждой стороны центрального слова
BATCH_SIZE = 64
EPOCHS = 5

# Preprocessing
def preprocess(data):
    tokenized = data.split()
    vocab = Counter(tokenized)
    word2index = {word: idx for idx, word in enumerate(vocab.keys())}
    index2word = {idx: word for word, idx in word2index.items()}
    return tokenized, word2index, index2word

def generate_cbow_pairs(tokenized, word2index, window_size):
    pairs = []
    for idx in range(window_size, len(tokenized) - window_size):
        context = [
            word2index[tokenized[i]]
            for i in range(idx - window_size, idx + window_size + 1)
            if i != idx
        ]
        center = word2index[tokenized[idx]]
        pairs.append((context, center))
    return pairs

# CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        # Суммирование контекстных эмбеддингов
        context_embeddings = self.embeddings(context_words).mean(dim=1)
        output = self.linear(context_embeddings)
        return output

# Training function
def train(data: str):
    tokenized, word2index, index2word = preprocess(data)
    vocab_size = len(word2index)
    cbow_pairs = generate_cbow_pairs(tokenized, word2index, WINDOW_SIZE)

    model = CBOW(vocab_size, EMBEDDING_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def create_batches(pairs, batch_size):
        random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            contexts = [pair[0] for pair in batch]
            centers = [pair[1] for pair in batch]
            yield torch.tensor(contexts, dtype=torch.long), torch.tensor(centers, dtype=torch.long)

    for epoch in range(EPOCHS):
        total_loss = 0
        for context_batch, center_batch in create_batches(cbow_pairs, BATCH_SIZE):
            # Forward pass
            predictions = model(context_batch)
            loss = criterion(predictions, center_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # Создаем словарь с эмбеддингами
    embeddings = model.embeddings.weight.detach().numpy()
    w2v_dict = {index2word[idx]: embeddings[idx] for idx in range(vocab_size)}
    return w2v_dict
