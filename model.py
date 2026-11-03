import torch
import torch.nn as nn
import math

from transformers_from_scratch.model import MultiHeadAttention


class InputEmbedding(nn.Module):
    def __init__(self,embedding_dim,vocab_size):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.embedding_layer=nn.Embedding(self.vocab_size, self.embedding_dim)
        
    def forward(self,x):
        return self.embedding_layer(x)*math.sqrt(self.embedding_dim)
    
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,max_sequence_length,dropout):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.max_sequence_length=max_sequence_length
        self.dropout=dropout
        
        matrix=torch.zeros([self.max_sequence_length,self.embedding_dim])
        
        position=torch.arange(0,self.max_sequence_length,dtype=torch.float64).unsqueeze(1)
        position_scores=torch.exp(torch.arange(0,self.embedding_dim,2).float()*(-1*math.log(10000.0)/self.embedding_dim))
        matrix[:,0::2]=torch.sin(position*position_scores)
        matrix[:,1::2]=torch.cos(position*position_scores)
        
        matrix=matrix.unsqueeze(0)
        self.register_buffer('positional_encoding', matrix)
        
        
    def forward(self,x):
        x=x+(self.positional_encoding[:,x.shape[1],:]).requires_grad_(False)
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
    
class ff_nn(nn.Module):
    def __init__(self,input_dimension,hidden_dimension,dropout):
        super().__init__()
        self.input_dimension=input_dimension
        self.hidden_dimension=hidden_dimension
        self.linear1=nn.Linear(input_dimension,hidden_dimension)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(hidden_dimension,input_dimension)
        
    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))  
    
    
class MultiheadAttention(nn.Module):
    
    def __init__(self,embedding_dim,nheads,dropout):
        self.embedding_dim=embedding_dim
        self.nheads=nheads
        self.dropout=nn.Dropout(dropout)
        assert self.embedding_dim%self.nheads==0,"Cannot divide the vector into given number of heads"
        self.embedding_div=self.embedding_dim/self.nheads
        
        self.w_q=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_k=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_v=nn.Linear(self.embedding_dim,self.embedding_dim)    
        
        self.w_o=nn.Linear(self.embedding_dim,self.embedding_dim)
        
    @staticmethod
    def find_attention(query,key,value,mask,dropout):
        
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask:
            attention_scores=attention_scores.masked_fill_(mask==0,-1e9)
            
        if dropout:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores


    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        
        query=query.view(query.shape[0],query.shape[1],self.nheads,self.embedding_div).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.nheads,self.embedding_div).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.nheads,self.embedding_div).transpose(1,2)
        
        x,attn=MultiHeadAttention.find_attention(query,key,value,mask,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.dim[0],-1,self.heads*self.embedding_div)
        
        return self.final_w(x)
        