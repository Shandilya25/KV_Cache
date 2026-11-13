import torch 
import torch.nn as nn
import math

from config import get_configuration
config=get_configuration()


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
    
class LayerNormalization(nn.Module):
    def __init__(self,features,eps):
        super().__init__()
        self.features=features
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
        
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps) + self.beta
class FeedForward(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout):
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.dropout=nn.Dropout(dropout)
        self.linea1=nn.Linear(embedding_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,embedding_dim)
        
    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class GroupedQueryAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,grouped_heads,dropout):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads
        self.grouped_heads=grouped_heads
        self.dropout=nn.Dropout(dropout)
        
        assert self.embedding_dim%self.num_heads==0,"Embedding dimension is not divisible by number of heads"
        assert self.embedding_dim%self.grouped_heads==0,"Embedding dimension is not divisible by number of grouped heads"
        
        self.head_dim=self.embedding_dim//self.num_heads
        self.group_dim=self.embedding_dim//self.grouped_heads
        
        self.w_q=nn.Linear(self.embedding_dim,self.head_dim*self.num_heads)
        self.w_k=nn.Linear(self.embedding_dim,self.head_dim*self.grouped_heads)
        self.w_v=nn.Linear(self.embedding_dim,self.head_dim*self.grouped_heads)
        
        self.w_o=nn.Linear(self.embedding_dim,self.embedding_dim)
        
        self.register_buffer('Key_Cache',torch.zeros(config['batch_size'],grouped_heads,config['sequence_length'],self.embedding_dim),persistent=False)
        self.register_buffer('Value_Cache',torch.zeros(config['batch_size'],grouped_heads,config['sequence_length'],self.embedding_dim),persistent=False)
        self.cache_len=0
        
    def forward(self,q,k,v,mask=None,use_cache=None):
        
                
        
        