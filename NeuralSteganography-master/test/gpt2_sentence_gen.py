import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel,GPT2Model
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
text='You'
indexed_tokens=tokenizer.encode(text)#对现有文本进行索引编码
tokens_tensor=torch.tensor([indexed_tokens])#编码转张量为输入
model=GPT2LMHeadModel.from_pretrained('gpt2')#调用gpt2模型
model.eval()
with torch.no_grad():
    for i in range(10):
        outputs = model(tokens_tensor)#将输入放入模型
        prediction = outputs[0]#获取下一词的条件概率分布
        print(prediction)
        predicted_index = torch.argmax(prediction[0, -1, :]).item()#选择最大概率的词的索引
        print(predicted_index)
        indexed_tokens=indexed_tokens+[predicted_index]
        print(indexed_tokens)
        predicted_text = tokenizer.decode(indexed_tokens)#将索引加入现在的索引编码后解码成文本
        print(predicted_text)
        tokens_tensor=torch.tensor([indexed_tokens])

    text = predicted_text  # 文本替换

