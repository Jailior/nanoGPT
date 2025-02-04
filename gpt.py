import torch
import torch.nn as nn
from torch.nn import functional as F

# -------
# Model architecture based on Attention is All You Need
# Deviations:
# - layer norm is applied before transformations in accordance to more
# modern GPTs
# ------


# hyperparameters
batch_size = 32
block_size = 8
epochs = 5000
eval_interval = 500 
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ---------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# list of characters which occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# turn text in encoded tensor
data = torch.tensor(encode(text), dtype=torch.long)

# training and test splits
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

# get batch of batch size from data split
def get_batch(split):
  data = train_data if split == 'train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

# estimate losses over interval eval_iters
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'test']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


class Head(nn.Module):
  """Single head self attention block"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)
    # compute attention scores
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) ==> (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    # weighted aggregation
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  """Multiple self attention heads in parallel, a multi head attention block"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)

  def forward(self, x):
    out  = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out
  

class FeedForward(nn.Module):
  """a simple linear layer followed by a ReLU"""

  def __init__(self, n_embd):
    super().__init__()
    self.net =  nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd)
    )

  def forward(self, x):
    return self.net(x)
  

class TransformerBlock(nn.Module):
  """ A transformer block from Attention Is All You Need"""

  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sa_heads = MultiHeadAttention(n_heads, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    x = self.ln1(x)
    x = x + self.sa_heads(x)
    x = self.ln2(x)
    x = x + self.ffwd(x)
    return x


class GPTLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
      TransformerBlock(n_embd, 4),
      TransformerBlock(n_embd, 4),
      TransformerBlock(n_embd, 4),
      TransformerBlock(n_embd, 4)
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    
    token_embeds = self.token_embedding_table(idx) # (B,T,C) batch size, context length, n_embd
    pos_embeds = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = token_embeds + pos_embeds
    x = self.blocks(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):  
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
  

model = GPTLanguageModel()
m = model.to(device)

# creating optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

# training and testing loop
for epoch in range(epochs):
  
  # evaluate and print train and test losses
  if epoch % eval_interval == 0 or epoch == epochs - 1:
    losses = estimate_loss()
    print(f"epoch: {epoch} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")
  
  # sample the training data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate from model
print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))