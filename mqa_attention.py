import torch 
import torch.nn as nn
import math

from config import get_configuration

config=get_configuration()
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
    
class FeedForward(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.hidden_dim=hidden_dim
        self.dropout=nn.Dropout(dropout)
        self.linear1=nn.Linear(embedding_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,embedding_dim)
        
    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiQueryAttention(nn.Module):
    def __init__(self,embedding_dim,dropout,n_heads):
        super().__init()
        self.embedding_dim=embedding_dim
        self.dropout=nn.Dropout(dropout)
        self.n_heads=n_heads
        
        assert self.embedding_dim%self.n_heads==0,"Cannot divide the vector into given number of heads"
        self.embedding_div=self.embedding_dim//self.n_heads
        
        self.w_q=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_k=nn.Linear(self.embedding_dim,self.embedding_div)
        self.w_k=nn.Linear(self.embedding_dim,self.embedding_div)
        
        self.w_o=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.register_buffer('key_cache',torch.zeros(config["batch_size"],1,config["seq_len"],self.embedding_div),persistent=False)
        self.register_buffer('value_cache',torch.zeros(config["batch_size"],1,config["seq_len"],self.embedding_div),persistent=False)
        self.cache_len=0
        
    def forward(self,q,k,v,mask=None,use_cache=False):
        batch,tokens,_=q.shape
        
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        
        query=query.view(query.shape[0],query.shape[1],self.n_heads,self.embedding_div).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],1,self.embedding_div).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],1,self.embedding_div).transpose(1,2)
        
        if use_cache:
            start=self.cache_len
            end=start+tokens
            assert end<config["seq_len"],"Cache length exceeded the maximum sequence length"
            self.key_cache[:batch,:,start:end,:]=key
            self.value_cache[:batch,:,start:end,:]=value
            self.cache_len=end
            key=self.key_cache[:batch,:,:self.cache_len,:]
            value=self.value_cache[:batch,:,:self.cache_len,:]  
            
            if mask is not None:

                mask=mask[:,:,start:end,:end]

        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores=attention_scores.masked_fill(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        attention_scores=self.dropout(attention_scores)

        x=(attention_scores @ value)
        x=x.transpose(1,2).contiguous().view(batch,new_tokens,self.embedding_dim)

        return self.w_o(x)
            
        
        
        
        
        
        