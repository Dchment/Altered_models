import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel,GPT2Model
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
text='I choose to embrace the light,'
indexed_tokens=tokenizer.encode(text)#对现有文本进行索引编码(使用分词器）
tokens_tensor=torch.tensor([indexed_tokens])#编码转为张量来作为模型输入
predicted_index=0
predicted_text=''
model=GPT2LMHeadModel.from_pretrained('gpt2')#调用gpt2模型
model.eval()
with torch.no_grad():
    while(tokenizer.decode(predicted_index)!=('.' or '!' or '"' or '?') and len(text)<100):
        outputs = model(tokens_tensor)#将输入放入模型
        print(outputs[0].shape)
        prediction = outputs[0]#获取下一词的条件概率分布
        predicted_index = torch.argmax(prediction[0, -1, :]).item()#选择最大概率的词的索引
        indexed_tokens=indexed_tokens+[predicted_index]
        predicted_text = tokenizer.decode(indexed_tokens)#将索引加入现在的索引编码后解码成文本
        tokens_tensor=torch.tensor([indexed_tokens])
        text = predicted_text
    
    print(text) # 文本替换
    

