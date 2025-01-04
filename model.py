# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        
        # Store hidden dimension as instance variable
        self.hidden_dim = hidden_dim
        
        # Word Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # vocab_size: number of unique words in vocabulary
        # embedding_dim: each word is converted to a vector of size embedding_dim (128)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            embedding_dim,     # Input size: size of word embeddings
            self.hidden_dim,   # Hidden size: 256
            num_layers=2       # Using 2 LSTM layers stacked together
        )
        
        # Output Layer
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        # Transforms LSTM output back to vocabulary size for word prediction

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        
        embeds = self.embeddings(input)
        # Print shape of embeddings: [seq_len=124, batch_size, embedding_dim]
        # print(f"Embeddings shape: {embeds.shape}")
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))
    
        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        # Print shapes for debugging
        # print(f"Output shape: {output.shape}")  # Should be [seq_len * batch_size, vocab_size]
        # print(f"Hidden state shape: {hidden[0].shape}")  # Should be [2, batch_size, hidden_dim] 
        # print(f"Cell state shape: {hidden[1].shape}")  # Should be [2, batch_size, hidden_dim]
        return output, hidden
