import torch
import torch.nn as nn
import torch.nn.functional as F
import Configure
import Huffman_Encoding
import Model
import DataProcess_v2
import numpy as np
import collections
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
class Steganography():
    '''
    file = './data/movie.txt'
    word_all = []
    data = open(file, 'r', encoding='utf-8').readlines()  # 读取数据
    for line in range(len(data)):
        data_line = data[line]
        line_words = data_line.split(' ')
        for word in line_words:
            word_all.append(word)  # 所有词列表
    words = list(set(word_all))  # char vocabulary词汇表（无重复单词）

    data_size, _vocab_size = len(word_all), len(words)
    print('data has %d words, %d unique.' % (data_size, _vocab_size))
    word_to_idx = {wo: i for i, wo in enumerate(words)}  # 单词与索引一一对应的字典
    idx_to_word = {i: wo for i, wo in enumerate(words)}  # 索引与单词一一对应的字典
    '''
    def __init__(self,dataset):
        self.start_words, self.sentences = DataProcess_v2.data_read('data/tweet.txt')
        self.data, self.labels, self.embed_arr, self.data_dict = DataProcess_v2.prepare_data(self.sentences, Configure.Config)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        class TextData(Dataset):#构建样本与标签的dataset
            def __init__(self, dataset,labels):
                self.data = dataset
                self.label=labels

            def __getitem__(self, item):
                return self.data[item],self.label[item]

            def __len__(self):
                return len(self.data)
        self.dataloader = DataLoader(dataset=TextData(self.data,self.labels), batch_size=Configure.Config.batch_size, shuffle=True,
                                collate_fn=self.padding)#用dataset构建dataloader

    def padding(self,batch):#对dataloader的每个batch进行补齐，方便构建tensor
        data=[i[0] for i in batch]
        label=[i[1] for i in batch]
        #pad_sequence用于将tensor数组中的每个tensor元素补齐到同一长度
        data_batch=pad_sequence([torch.from_numpy(np.array(line)) for line in data],batch_first=True,padding_value=0).long()
        label_batch=pad_sequence([torch.from_numpy(np.array(line)) for line in label],batch_first=True,padding_value=0).long()
        #print(data_batch.shape)
        #print(label_batch.shape)
        return data_batch,label_batch


    def pro_start_word(self,statistics1):  # 使用关键词列表随机选择句子开头
        sel_word_sta = []  # 各开头词
        sel_value_sta = []  # 各开头词使用的概率
        for i in range(100):
            k = statistics1[i]
            key = k[0]
            value = k[1]
            sel_word_sta.append(key)
            sel_value_sta.append(value)
        sel_value_sta = np.array(sel_value_sta)
        sel_value_sta = sel_value_sta / float(sum(sel_value_sta))
        start = np.random.choice(sel_word_sta, 1, p=sel_value_sta)
        start_word = start[0]
        while not start_word.islower():
            start = np.random.choice(sel_word_sta, 1, p=sel_value_sta)
            start_word = start[0]
        return start_word

    def train(self):#训练样本
        model = Model.model(Configure.Config)
        model.to(self.device)
        for iter in range(Configure.Config.iteration):
            for batch_num,batch in enumerate(self.dataloader):
                #print(batch)
                inputs=batch[0]
                label=batch[1]
                inputs=inputs.to(self.device)
                label=label.to(self.device)
                loss=0
                print(inputs.shape)
                outputs = model(inputs)
                outputs=outputs.transpose(0, 1)
                outputs=outputs.transpose(1, 2)
                #print(outputs)
                #print(outputs.shape)
                criterion = torch.nn.CrossEntropyLoss()
                loss=criterion(outputs,label)
                if (batch_num % 100 == 0):
                    print("epoch: %d, batch: %d, loss : %f" % (iter, batch_num, loss))
                optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (batch_num % 10000 == 0 and batch_num!=0):
                    torch.save(model,
                               Configure.Config.model_path + Configure.Config.chosen_dataset + '/' + Configure.Config.chosen_dataset + '_b' + str(
                                   batch_num) + '.pkl')
            torch.save(model,Configure.Config.model_path + Configure.Config.chosen_dataset + '/' + Configure.Config.chosen_dataset + '.pkl')


    def hiding(self):#进行隐写并输出隐写文本
        bit_num=2
        bit_num=np.int32(bit_num)#每次生成的词所嵌入的比特数
        index=1
        statistics = collections.Counter(self.start_words)
        statistics1 = sorted(statistics.items(), key=lambda item: item[1], reverse=True)
        count=0
        bit_stream = open(os.path.dirname(os.path.realpath(__file__)) + '/bit_stream/bit_stream.txt', 'r').readline()
        outfile = open('./generate/'+Configure.Config.chosen_dataset+'/'+Configure.Config.chosen_dataset +'_'
                       +str(bit_num) + 'bit' + '_' + str(index) + '.txt', 'w')
        bitfile = open('./generate/'+Configure.Config.chosen_dataset+'/'+Configure.Config.chosen_dataset +'_'
                       +str(bit_num) + 'bit' + '_' + str(index) + '.bit', 'w')
        bit_index=0
        model=torch.load(Configure.Config.model_path+Configure.Config.chosen_dataset+'/'+Configure.Config.chosen_dataset+'.pkl')
        bit=""
        while count<100:
            start_word = self.pro_start_word(statistics1)#随机选择一个起始词
            if start_word == 'unknown':
                continue
            start_idx = self.data_dict.word2idx[start_word]  # 起始词标签
            gen_stegotext = ['<BOS>']  # 生成的隐写文本
            gen_stegotext.append(start_word)
            input_data=torch.LongTensor([[start_idx]])     
            gen = self.data_dict.word2idx['unknown']
            for i in range(Configure.Config.len_of_generation-2):
                input_data=input_data.to(self.device)
                if self.data_dict.idx2word[gen] in ['\n', '','<EOS>']:  # 如果下一个词是空格或换行符，结束此短语，换行
                    break
                model=model.to(self.device)
                output = model(input_data)
                prob=output.reshape(-1)#模型生成的概率分布
                print(prob)
                prob_sort = sorted(prob)
                prob_sort.reverse()  # 按词被选中的概率逆序排列
                word_prob = [prob_sort[i] for i in range(2 ** int(bit_num))]
                print(word_prob)
                prob = prob.tolist()
                words_prob = [(prob.index(word_prob[i]), word_prob[i]) for i in range(2 ** int(bit_num))]  # 每个词及其被选中的概率
                nodes = Huffman_Encoding.createNodes([item[1] for item in words_prob])  # 哈夫曼树的全部叶子结点，也就是各词
                root = Huffman_Encoding.createHuffmanTree(nodes)  # 创建哈夫曼树
                codes = Huffman_Encoding.huffmanEncoding(nodes, root)  # 哈夫曼树编码
                bit=""
                for i in range(2 ** int(bit_num)):  # 将对应的比特流嵌入词中，比特流最大长度为2^输入值
                    if bit_stream[bit_index:bit_index + i + 1] in codes:
                        code_index = codes.index(bit_stream[bit_index:bit_index + i + 1])  # 找出比特流可以对应的单词
                        gen = words_prob[code_index][0]  # 单词的索引
                        input_data = torch.LongTensor([[np.int32(gen)]])  # 此单词变为下一个用于预测单词的测试集
                        gen_stegotext.append(self.data_dict.idx2word[gen])  # 将单词添加进隐写文本中
                        print(gen_stegotext)
                        if self.data_dict.idx2word[gen] in ['\n', '','<EOS>']:
                            break
                        bit += bit_stream[bit_index: bit_index + i + 1]
                        bit_index = bit_index + i + 1
                        break
            if len(gen_stegotext) < 5:
                continue
            gen_sen = ' '.join([word for word in gen_stegotext if word not in ['<BOS>',"\n", "",'<EOS>']])
            count+=1
            outfile.write(gen_sen + "\n")
            bitfile.write(bit)







s=Steganography('data/tweet.txt')
s.hiding()







