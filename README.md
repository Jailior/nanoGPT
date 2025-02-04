# nanoGPT
A simple GPT model based on the Attention Is All You Need paper

## About
nanoGPT is a small scale generative pre-trained transformer model, with ~11M parameters, trained on the *Tiny Shakespeare* dataset
based on the transformer shown in the 2017 paper, Attention Is All You Need. This model focuses on the generation of Shakespeare-like text 
based on no inputs.

The model is able to reach a 1.4956 test loss after 15 minutes of training. With
your eyes squinted the text it generates may look like Shakespeare.

Sample text generated:
```
LEONTES:
At first cost Cali:
Either in our wallow general; and, conceiler,
I presently ax hime of my warrantly perfect,
As to live leage in draw as go. A Camillo wifed
this path for men's yet, but may calall bind
redeed, you lie with some mans, as see you
As lessed for her wasquit with a back
With a liege. Your fair missalies say.

SICINIUS:
Well you but yest? it is At was for trial
The norbones build by the general all
Tir Rome of the prace o' the put villain statute'h!
```

## Usage
Git clone the repository, or only download `gpt_model.pth` and the runnable script `run.py`.

Use PyTorch to load the state dictionary from `gpt_model.pth` or use `run.py` altering this line:
```
output = decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist())
```
Changing `idx` to an encoded string of your choice and `max_new_tokens` to adjust the number of characters
in the output.

## Dependencies
- pytorch
- numpy

Installation:
```
pip install torch numpy
```

## Notes

The transformer only involves the decoder as the transformer in the original paper
was intended for language translation. In this application of the architecture our inputs do not
need to be encoded in the transformer.

Another thing to note is that the layer norm in the original paper is applied after multi-head
attention and feedforward blocks but for this model it is applied before as this is what more 
modern GPTs do as well.
