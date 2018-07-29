
# Tutorial of implementing Elmo with PyTorch

- Author:  _YuriAntonovsky@QuantumAgent_


```python
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook,tqdm_pandas
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
lmap=lambda func,it: list(map(lambda x:func(x),it))
```


```python
import spacy
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
n = lambda x: nlp(x, disable=['tagger', 'ner', 'textcat', 'parser'])
```

##  Preprocessing

- (one-hot)tize characters
- build vocabulary


```python
titles=pd.read_csv('data/title_text.csv').dropna()
```


```python
characters=defaultdict()
characters.setdefault('',len(characters))
for c in " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}":
    characters.setdefault(c,len(characters))
```


```python
titles['date']=titles['date'].astype(np.datetime64)
```

    /usr/local/lib/python3.5/dist-packages/pandas/core/internals.py:3462: FutureWarning: Passing in 'datetime64' dtype with no frequency is deprecated and will raise in a future version. Please pass in 'datetime64[ns]' instead.
      return self.apply('astype', dtype=dtype, **kwargs)



```python
titles.drop('Unnamed: 0',axis=1,inplace=True)
```


```python
data=list(titles['title'])
```


```python
data=lmap(lambda x:x.strip().lower(),data)
```


```python
vocabulary = defaultdict()
vocabulary.setdefault('',len(vocabulary))
vocabulary.setdefault('<SOS>',len(vocabulary))
vocabulary.setdefault('<EOS>',len(vocabulary))
preprocessed_data=[]
title_date=[]
for i,d in tqdm_notebook(enumerate(data)):
    if np.array(lmap(lambda x:x in characters ,list(d))).prod() == 0: continue
    dn=n(d)
    tokens=lmap(lambda x:x.text,dn)
    word_index=[]
    chars=[]
    word_index.append(vocabulary['<SOS>'])
    for t in tokens:
        t_text=t.strip()
        char_index=[]
        vocabulary.setdefault(t_text, len(vocabulary))
        word_index.append(vocabulary[t_text])
        for c in list(t_text):
            char_index.append(characters[c])
        chars.append(char_index)
    word_index.append(vocabulary['<EOS>'])
    preprocessed_data.append((word_index,chars))
    title_date.append(titles['date'].iloc[i])
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in Jupyter Notebook or JupyterLab, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another notebook frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    



```python
title_date=np.array(title_date)

np.save('./data/title_date',title_date)
```


```python
len(vocabulary)
```




    94297




```python
len(characters)
```




    70




```python
with open('./data/vocabulary.pkl','wb+') as f:
    pickle.dump(vocabulary,f)

with open('./data/characters.pkl','wb+') as f:
    pickle.dump(characters,f)
```


```python
corpus_tokens=lmap(lambda x:x[0],preprocessed_data)

corpus_chars=lmap(lambda x:x[1],preprocessed_data)
```


```python
assert (lmap(lambda x:len(x),corpus_tokens) ==lmap(lambda x:len(x)+2,corpus_chars))
```


```python
max_char_length=max(lmap(lambda y:max(lmap(lambda x:len(x),y)),corpus_chars))

max_token_length=max(lmap(lambda x:len(x),corpus_tokens))
```


```python
# construct one-hot embedding
onehot=np.eye(len(characters))
onehot[0,0]=0
```


```python
def crop_pad(max_leng, word_index):
    if len(word_index) > max_leng:
        return word_index[:max_leng]
    pad_leng = max_leng - len(word_index)
    word_index = word_index + [0] * pad_leng
    assert len(word_index) == max_leng
    return word_index
```


```python
pad_tokens=[]
for t in tqdm_notebook(corpus_tokens):
    pad_tokens.append(crop_pad(max_leng=max_token_length,word_index=t))
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in Jupyter Notebook or JupyterLab, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another notebook frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    



```python
pad_tokens=np.array(pad_tokens)

corpus_tokens=pad_tokens
```


```python
corpus_tokens.shape
```




    (432006, 40)




```python
pad_chars=[]
for s in tqdm_notebook(corpus_chars):
    pad_sent=[]
    for w in s:
        pad_sent.append(crop_pad(max_leng=max_char_length,word_index=w))
    if len(s)<max_token_length:
        pad_leng=max_token_length-len(s)
        pad_sent=pad_sent+([[0] * max_char_length] * pad_leng)
    pad_chars.append(pad_sent)
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in Jupyter Notebook or JupyterLab, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another notebook frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    



```python
pad_chars=np.array(pad_chars)

corpus_chars=pad_chars
```


```python
np.save('data/corpus_tokens',corpus_tokens)
```


```python
np.save('data/corpus_chars',corpus_chars)
```

## Character Embedding
- kernel_size: 2,3,4,5
- filters: 32,32,32,32
- word_vector_length: $\sum{filters}=128$


```python

class CharacterEmbedding(nn.Module):
    def __init__(self, weight_matrix, dropout, filters=[32, 32, 32, 32], kernel_sizes=[2, 3, 4, 5]):
        super(CharacterEmbedding, self).__init__()
        input_size = weight_matrix.shape[1]
        self.output_size = sum(filters)
        self.embedding = nn.Embedding(input_size, input_size, _weight=weight_matrix)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_size, out_channels=filters[i], kernel_size=(1, kernel_sizes[i])) for i in range(len(filters))])
        self.highway_h = nn.Linear(sum(filters), sum(filters))
        self.highway_t = nn.Linear(sum(filters), sum(filters))
        self.dropout = dropout
    
    def conv_and_pool(self, x, conv):
        b, w, c, e = tuple(x.shape)
        x_out = conv(x.view(b, e, w, c)).max(dim=-1)[0]
        return x_out.view(b, w, -1)
    
    def highway(self, x):
        x_h = F.relu(self.highway_h(x))
        x_t = F.sigmoid(self.highway_t(x))
        x_out = x_h * x_t + x * (1 - x_t)
        return x_out
    
    def forward(self, x, train=False):
        x = self.embedding(x)
        results = list(map(lambda conv: self.conv_and_pool(x, conv), self.convs))
        results = torch.cat(results, dim=-1)
        results = self.highway(results)
        if train:
            results = self.dropout(results)
        return results



```

## Bi-Language-Model
- #layer=2
- bidirectional LSTM
- hidden_size=64


```python
class BiLM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, word_number, dropout):
        super(BiLM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.out_fc = nn.Linear(2 * hidden_size, word_number)
        self.dropout = dropout
    
    def forward(self, x, hidden=None, train=False):
        out = x
        if train:
            out = self.dropout(out)
        out, (h_n, h_c) = self.lstm(out, hidden)
        out = F.log_softmax(F.relu(self.out_fc(out)), dim=-1)
        return out, (h_n, h_c)
```

##  Elmo
- use word vector $w_t$ as input where $w_t \in R^{128} $
- ${h_c}^{l}=[{h_{cf}}^{l};{h_{cb}}^{l}]$
- ${h_n}^{l}=[{h_{cf}}^{l};{h_{cb}}^{l}]$
- ${h}^{l}=[{h_n}^{l};{h_c}^l]$
- $h=\sum_l{s_l}{h^l}$
- $\sum_l{s_l}=1$


```python
class Elmo(object):
    def __init__(self, weight_matrix, word_number, hidden_size=64, num_layers=2, learning_rate=1e-3, dp=0.2):
        super(Elmo, self).__init__()
        self.dropout = nn.Dropout(p=dp)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.char_emb = CharacterEmbedding(weight_matrix=torch.tensor(weight_matrix, dtype=torch.float32), dropout=self.dropout)
        self.bilm = BiLM(input_size=self.char_emb.output_size, hidden_size=self.hidden_size, num_layers=self.num_layers, word_number=word_number, dropout=self.dropout)
        self.optimizer = optim.Adam(list(self.char_emb.parameters()) + list(self.bilm.parameters()), lr=learning_rate)
        self.criterion = nn.NLLLoss()
    
    def _encode(self, x, gamma=1):
        # weight vector of every layer not implemented!
        max_word_length = (((x > 0).sum(axis=-1)) > 0).sum(axis=1).max()
        x_w = x[:, :max_word_length, :]
        x_w = self.char_emb(torch.tensor(x_w), train=False)
        _, (hn, hc) = self.bilm(torch.tensor(x_w), train=False)
        hns = []
        hcs = []
        for i in range(0, self.num_layers * 2, 2):
            hns.append(torch.cat([hn[i, :, :], hn[i + 1, :, :]], dim=-1))
            hcs.append(torch.cat([hc[i, :, :], hc[i + 1, :, :]], dim=-1))
        # same weight for each layer
        hns = torch.stack(hns).mean(0)
        hcs = torch.stack(hcs).mean(0)
        
        encode_result = torch.cat([hns, hcs], dim=-1)
        return encode_result.detach().numpy()
    
    def encode(self, X, batch_size=64, gamma=1):
        pointer = 0
        results_ = np.zeros((1, self.hidden_size*4))
        while pointer < X.shape[0]:
            batch_x = X[pointer:(pointer + batch_size)]
            result_batch = self._encode(batch_x, gamma=gamma)
            results_ = np.concatenate((results_, result_batch))
            pointer += batch_size
        return results_[1:]
    
    def _train(self, x, y):
        max_word_length = (y > 0).sum(axis=1).max()
        batch_x = x[:, :max_word_length, :]
        batch_y = y[:, :max_word_length]
        self.optimizer.zero_grad()
        y_f = batch_y[:, 1:]
        y_b = batch_y[:, :-1]
        x_w = self.char_emb(torch.tensor(batch_x), train=True)
        y_hat, (_, _) = self.bilm(x_w, train=True)
        y_hat_f = y_hat[:, :-1, :]
        y_hat_b = y_hat[:, 1:, :]
        loss_f = self.criterion(y_hat_f.transpose(1, -1), torch.tensor(y_f))
        loss_b = self.criterion(y_hat_b.transpose(1, -1), torch.tensor(y_b))
        loss = (loss_f + loss_b) / 2
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, X, y, batch_size=64, epoch=2):
        global_step = 0
        for e in range(epoch):
            pointer = 0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            while pointer < X.shape[0]:
                batch_x = X_shuffled[pointer:(pointer + batch_size)]
                batch_y = y_shuffled[pointer:(pointer + batch_size)]
                mean_loss = self._train(batch_x, batch_y)
                print(mean_loss, 'batch%:', round((pointer / X.shape[0]) * 100, 4), 'epoch:', e)
                pointer += batch_size
#                 writer.add_scalar(tag='loss', scalar_value=mean_loss, global_step=global_step)
                global_step += 1
            self.save_model()
    
    def save_model(self, model_path='./ELMO'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        print("saving models")
        torch.save(self.char_emb, model_path + '/char_emb.pkl')
        torch.save(self.bilm, model_path + '/bilm.pkl')
    
    def load_model(self, model_path='./ELMO'):
        print("loading models")
        self.char_emb = torch.load(model_path + '/char_emb.pkl')
        self.bilm = torch.load(model_path + '/bilm.pkl')

```

## Experiment
- sample size: 128


```python
batch_x=corpus_chars[:128,:,:]

batch_y=corpus_tokens[:128,:]
```


```python
batch_x.shape
```




    (128, 40, 30)




```python
batch_y.shape
```




    (128, 40)




```python
elmo=Elmo(weight_matrix=onehot,word_number=len(vocabulary),learning_rate=1e-3)
```


```python
%%time
elmo.train(X=batch_x,y=batch_y,batch_size=64,epoch=2)
```

    11.464266777038574 batch%: 0.0 epoch: 0
    11.46330738067627 batch%: 50.0 epoch: 0
    saving models


    /usr/local/lib/python3.5/dist-packages/torch/serialization.py:193: UserWarning: Couldn't retrieve source code for container of type CharacterEmbedding. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /usr/local/lib/python3.5/dist-packages/torch/serialization.py:193: UserWarning: Couldn't retrieve source code for container of type BiLM. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


    11.452033996582031 batch%: 0.0 epoch: 1
    11.451211929321289 batch%: 50.0 epoch: 1
    saving models
    CPU times: user 1min, sys: 7.91 s, total: 1min 8s
    Wall time: 20.6 s



```python
elmo.load_model()
```

    loading models



```python
results=elmo.encode(X=batch_x,batch_size=64)
```

## Seems OK, Thank you
