import torch
import torch.nn.functional as F
import numpy as np
from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg,get_model, encode_context
from arithmetic import encode_arithmetic, decode_arithmetic
from transformers import GPT2LMHeadModel,GPT2Tokenizer
enc,model=get_model(model_name='gpt2')
text='hello'
e=enc.decode(1)
print(e.encode('utf-8'))