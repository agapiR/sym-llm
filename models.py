import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Basic Components
"""

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head=3, block_size=256, dropout=0.5):
        # n_embd: embedding dimension 
        # n_head: the number of heads we'd like
        # block_size:   ?
        # dropout:      ?     
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

"""
Transformer
"""

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        block_list = [Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*block_list)
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

        # store device for forward pass
        self.device = device

        # store block size for generation
        self.block_size = block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    


"""
Symbolic Components
"""

class SymbolicAttentionHead(nn.Module):
    """ one head of symbolic self-attention """

    def __init__(self, cooc, h, self_loops=True):
        super().__init__()
        # cooc statistics
        vocab_size,_,n_heads = cooc.shape
        self.h_cooc = cooc[:,:,h]
        # global representation (Value matrix)
        self.value = nn.Embedding(vocab_size, vocab_size*n_heads)
        symbolic_token_embeddings = torch.flatten(cooc, start_dim=1, end_dim=2) # combine dimensions 1 and 2 into one
        self.value.load_state_dict({'weight': symbolic_token_embeddings})
        self.value.weight.requires_grad = False
        # self loop in context-aware representation
        self.add_self_loops = self_loops

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # T: time-steps
        # N: vocab size
        # K: number of heads
        # head size (hs): N*K
        device = x.device
        B,T,C = x.shape
        k = x   # (B,T,1)
        q = x   # (B,T,1)
        # Compute the symbolic self-attention scores 
        att_scores = torch.empty(B, T, T) #.to(device)
        # TODO: For now, we just do calculate attention scores per batch. Not the most efficient way.
        for b in range(B):
            # compute attention scores ("affinities") for batch
            token_pairs = torch.cartesian_prod(k[b].squeeze(), q[b].squeeze())
            ind = token_pairs[:,0] * self.h_cooc.size(1) + token_pairs[:,1]
            wei = torch.take(self.h_cooc, ind).reshape((T,T)) # (T, T)
            wei = F.softmax(wei, dim=-1).unsqueeze(0) # (1, T, T)
            att_scores[b] = wei
            # the representation of the token itself is added, weighted by 1.
            if self.add_self_loops:
                self_loops = torch.eye(T, T) #.to(device)
                att_scores[b] += self_loops
        # Perform the weighted aggregation of the values
        v = self.value(x.squeeze()) # (B,T,hs)
        # Compute the context-aware representations
        out = att_scores @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class SymbolicMultiHeadAttention(nn.Module):
    """ multi-head symbolic self-attention """

    def __init__(self, n_embd, cooc, dropout):
        super().__init__()
        vocab_size,_,num_heads = cooc.shape
        head_size = vocab_size*num_heads
        print("From SymbolicMultiHeadAttention: N {}, K {}, hs {}, embedding_dim {}".format(vocab_size, num_heads, head_size, n_embd))
        self.heads = nn.ModuleList([SymbolicAttentionHead(cooc, h) for h in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class SymbolicBlock(nn.Module):
    """ Symbolic Transformer block: symbolic communication followed by computation """

    def __init__(self, n_embd, cooc, dropout=0.5):
        # n_embd: embedding dimension 
        # h_cooc: multi-head co-occurence statistics
        super().__init__()
        self.sa = SymbolicMultiHeadAttention(n_embd, cooc, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(self.ln(x))
        return x

class SymbolicMultiHeadAttentionEncoding(nn.Module):
    """ multi-head symbolic self-attention for encoding, no learned components """

    def __init__(self, cooc, self_loops=True):
        super().__init__()
        _,_,num_heads = cooc.shape
        self.heads = nn.ModuleList([SymbolicAttentionHead(cooc, h, self_loops=self_loops) for h in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, hs*T)
        return out


"""
Symbolic Transformer
"""

class SymGPTLanguageModel(nn.Module):

    def __init__(self, out_vocab_size, n_embd, n_head, n_layer, block_size, dropout, device, cooc):
        super().__init__()
        block_list = [SymbolicBlock(n_embd, cooc, dropout=dropout)] + [Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer-1)]
        self.blocks = nn.Sequential(*block_list)
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, out_vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

        # store device for forward pass
        self.device = device

        # store block size for generation
        self.block_size = block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad: # we do not want to re-initialize the frozen embedding
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        # x = tok_emb + pos_emb # (B,T,C)
        x = idx.reshape((B, T, 1)) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

class SymLMEncoder(nn.Module):

    def __init__(self, out_vocab_size, device, cooc, self_loops=True):
        super().__init__()
        # symbolic module that produces a contex-aware representation
        self.sym_attn = SymbolicMultiHeadAttentionEncoding(cooc, self_loops=self_loops)
        # TODO: implement encoder
        # encoder module that maps the contex-aware representation into a single symbol
        # self.encoder = 
        # store device for forward pass
        self.device = device

    def forward(self, idx):
        B, T = idx.shape
        x = idx.reshape((B, T, 1)) # (B,T,C)
        # produce context aware intermediate representation
        x = self.sym_attn(x) # (B,T,C)
        # focus only on the last time step, 
        # the last token uses context from the whole context window/block
        x_rep = x[:, -1, :] # becomes (B, C)
        # encode the intermediate representation into a new symbol
        # TODO: encoder not implemented
        # x_enc = self.encoder(x)
        x_enc = 0
        return x_rep, x_enc
    


"""
Baseline
"""

class SimpleBlock(nn.Module):
    """ Symbolic Transformer block: symbolic communication followed by computation """

    def __init__(self, n_embd, dropout=0.5):
        # n_embd: embedding dimension 
        super().__init__()
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.ffwd(self.ln(x))
        return x
    

class FeedForwardSeqModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        block_list = [SimpleBlock(n_embd, dropout=dropout) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*block_list)
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

        # store device for forward pass
        self.device = device

        # store block size for generation
        self.block_size = block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx