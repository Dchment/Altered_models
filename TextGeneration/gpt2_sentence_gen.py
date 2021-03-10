import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import Huffman_Encoding
import utils
import bitarray

class Steganography():
    context = 'Our target'
    text_bit=''
    tokenizer,model=utils.get_model(context)
    def encode(self,tokenizer=tokenizer,model=model,context=context,start_point=0):
        start=start_point
        predicted_index=0
        indexed_tokens = tokenizer.encode(context)  # 对现有文本进行索引编码(使用分词器）
        tokens_tensor = torch.tensor([indexed_tokens])  # 编码转为张量来作为模型输入
        targettext=''
        text = 'this is secret!'
        ba = bitarray.bitarray()
        ba.frombytes(text.encode('utf-8'))
        bit = ba.tolist(True)  # 转换成每8位代表一个字符的比特流
        global str
        bit=[str(i) for i in bit]
        bit="".join(bit)
        print(bit)
        print(len(bit))
        with torch.no_grad():
            while (start<len(bit) or utils.quotataion_end(context+targettext)):
                outputs = model(tokens_tensor)  # 将输入放入模型
                prediction = outputs[0]  # 获取下一词的条件概率分布
                probability = prediction[0, -1, :]


                # 设置备选池，进行哈夫曼编码
                probability_s = F.softmax(probability, dim=0)
                candidate_pool = probability_s.topk(k=9)  # 备选池，其大小与编码长度有关
                print(candidate_pool)
                node_list = (Huffman_Encoding.createNodes(candidate_pool[0].numpy()))  # 需要进行编码的各单词概率集
                tree = Huffman_Encoding.createHuffmanTree(node_list)  # 哈夫曼树
                codes = Huffman_Encoding.huffmanEncoding(node_list, tree)  # 编码完成后的单词备选池
                print(codes)
                if (start < len(bit)):
                    index, str = utils.bit_same(codes, bit, start)  # 查询备选池中单词编码与秘密信息比特流的重合部分，并挑出来作为按照秘密信息挑选的最大概率词的索引
                    start += len(str)
                else:
                    index = torch.argmax(candidate_pool[0])

                predicted_index = candidate_pool[1][index].item()  # 选择最大概率的词的索引
                print(predicted_index)
                '''
                indexed_tokens = indexed_tokens + [predicted_index]
                predicted_text = tokenizer.decode(indexed_tokens)  # 将索引加入现在的索引编码后解码成文本
                tokens_tensor = torch.tensor([indexed_tokens])
                '''
                predicted_text=tokenizer.decode(predicted_index)
                targettext += predicted_text  # 文本替换
                print(context+targettext)
                indexed_tokens = indexed_tokens + [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens])
        self.text_bit=bit
        return context+targettext

    def decode(self,text,tokenizer=tokenizer,model=model,context=context):
        message=''
        predicted_index=0
        indexed_tokens = tokenizer.encode(context)  # 对现有文本进行索引编码(使用分词器）
        tokens_tensor = torch.tensor([indexed_tokens])  # 编码转为张量来作为模型输入
        text_indice=tokenizer.encode(text)
        start_point=len(tokenizer.encode(context))
        targettext = ''
        with torch.no_grad():
            for i in text_indice[start_point:]:
                outputs = model(tokens_tensor)  # 将输入放入模型
                prediction = outputs[0]  # 获取下一词的条件概率分布
                probability = prediction[0, -1, :]

                # 设置备选池，进行哈夫曼编码
                probability_s = F.softmax(probability, dim=0)
                candidate_pool = probability_s.topk(k=9)  # 备选池，其大小与编码长度有关
                node_list = (Huffman_Encoding.createNodes(candidate_pool[0].numpy()))  # 需要进行编码的各单词概率集
                tree = Huffman_Encoding.createHuffmanTree(node_list)  # 哈夫曼树
                codes = Huffman_Encoding.huffmanEncoding(node_list, tree)  # 编码完成后的单词备选池
                print(candidate_pool)
                print(codes)
                for j in range(len(candidate_pool[1])):
                    if i==candidate_pool[1][j].item():
                        message+=codes[j]
                        predicted_index=candidate_pool[1][j].item()
                        start_point +=len(codes[j])
                        break
                predicted_text=tokenizer.decode(predicted_index)
                targettext += predicted_text  # 文本替换
                print(context+targettext)
                indexed_tokens = indexed_tokens + [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens])
        return message













            






