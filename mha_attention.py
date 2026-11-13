import torch 
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self,embedding_dim,vocab_size):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.embedding_dim)
    
class PositionalEmbedding(nn.Module):
    def __init__(self,embedding_dim,max_sequence_length,dropout):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.max_sequence_length=max_sequence_length
        self.dropout=nn.Dropout(dropout)
        
        #creating positional embedding matrix
        matrix=torch.zeros([self.max_sequence_length,self.embedding_dim])
        
        positions=torch.arange(0,self.max_sequence_length,dtype=torch.float64).unsqueeze(1)
        position_scores=torch.exp(torch.arange(0,self.embedding_dim,2).float()*(-1*math.log(1000.0)/self.embedding_dim))
        matrix[:,0::2]=torch.sin(positions*position_scores)
        matrix[:,1::2]=torch.cos(positions*position_scores)
        
        matrix=matrix.unsqueeze(0)
        self.register_buffer('positional_encoding', matrix)
        
    def forward(self,x):
        x=x+self.position_encoding[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)