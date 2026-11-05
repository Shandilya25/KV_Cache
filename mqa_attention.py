import torch 
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.embedding_layer=nn.Embedding(vocab_size,embedding_dim)
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.embedding_dim)
    
class Positional_embedding(nn.Module):
    def __init__(self,embedding_dim,vocab_size,max_sequence_length,dropout):
        super().__init()
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.max_sequence_length=max_sequence_length
        self.dropout=nn.Dropout(dropout)
        
        matrix=torch.zeros([self.max_sequence_length,self.embedding_dim])
        
        position=torch.arange(0,self.max_sequence_length,dtype=torch.float64).unsqueeze(1)
        scores=torch.exp(torch.arange(0,self.embedding_dim,2).float()*(-1*math.log(1000.0)/self.embedding_dim))
        matrix[:,0::2]=torch.sin(position*scores)
        matrix[:,1::2]=torch.cos(position*scores)
        
        matrix=matrix.unsqueeze(0)
        self.register_buffer('positional_encoding', matrix)
        
    def forward(self,x):
        x=x+self.positional_encoding[:,x.shape[1],:].requires_grad_(False)
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    def __init__(self,eps,features):
        super().__init__()
        self.eps=eps
        self.gamma=torch.zeros(features)
        self.beta=torch.ones(features)
        
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps) + self.beta
    