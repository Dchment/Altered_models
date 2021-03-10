import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel
def bit_same(strlist,string,start):
    for i in range(len(strlist)):
        st=string[start:start + len(strlist[i])]
        s=''.join(str(i) for i in st)
        if strlist[i]==s:
            return i,strlist[i]

def get_model(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # 调用gpt2分词器
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # 调用gpt2模型
    model.eval()
    return tokenizer,model

def quotataion_end(text):
    if text=='':
        return True
    elif text[-1]!=('.' or '?' or '!' or ')' or '"'):
        return True
    #elif text.count('"')%2!=0:
        #return True
    else:
        return False