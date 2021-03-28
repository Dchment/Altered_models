import torch
import torch.nn as nn
import torch.nn.functional as F
import Configure
import Huffman_Encoding
import Model
import DataProcess
import numpy as np
import collections
import gensim
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
        self.start_words, self.sentences = DataProcess.data_read(dataset)
        self.data, self.labels, self.embed_arr, self.data_dict=DataProcess.prepare_data(self.sentences, Configure.Config)

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

    def train(self):
        model = Model.model(Configure.Config)
        batch_num=len(self.data)//Configure.Config.batch_size
        for iter in range(Configure.Config.iteration):
            for b in range(batch_num):
                #print("batch :"+str(b))
                batch = torch.LongTensor(DataProcess.padding(self.data[b*Configure.Config.batch_size:(b+1)*Configure.Config.batch_size]))
                label = torch.LongTensor(DataProcess.padding(self.labels[b*Configure.Config.batch_size:(b+1)*Configure.Config.batch_size]))
                loss=0
                inputs = batch
                #print(inputs.shape)
                outputs = model(inputs)
                #print(outputs)
                outputs=outputs.transpose(0, 1)
                # inputs.resize([model.batch_size,model.vocab_size])
                criterion = torch.nn.CrossEntropyLoss()
                for i in range(len(outputs[0])):
                    loss+=criterion(outputs[i,:], label[i])
                loss=loss/Configure.Config.batch_size
                if(b%10==0):
                    print("epoch: %d, batch: %d, loss : %f" % (iter,b,loss))
                optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(b%1000==0):
                    torch.save(model,
                               Configure.Config.model_path + Configure.Config.chosen_dataset + '/' + Configure.Config.chosen_dataset +'_b'+str(b)+'.pkl')
            torch.save(model,
                       Configure.Config.model_path + Configure.Config.chosen_dataset + '/' + Configure.Config.chosen_dataset + '.pkl')


    def hiding(self):
        bit_num=2
        bit_num=np.int32(bit_num)
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
        model=torch.load(Configure.Config.model_path+Configure.Config.chosen_dataset+'/'+Configure.Config.chosen_dataset+'_b1000.pkl')
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
                if self.data_dict.idx2word[gen] in ['\n', '','<EOS>']:  # 如果下一个词是空格或换行符，结束此短语，换行
                    break
                output = model(input_data)
                prob=output.reshape(-1)
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
                        test_data = np.int32(gen)  # 此单词变为下一个用于预测单词的测试集
                        gen_stegotext.append(self.data_dict.idx2word[gen])  # 将单词添加进隐写文本中
                        print(gen_stegotext)
                        if self.data_dict.idx2word[gen] in ['\n', '','<EOS>']:
                            break
                        bit += bit_stream[bit_index: bit_index + i + 1]
                        bit_index = bit_index + i + 1
                        break
            if len(gen_stegotext) < 5:
                continue
            gen_sen = ' '.join([word for word in gen_stegotext if word not in ["\n", "",'<EOS>']])
            count+=1
            outfile.write(gen_sen + "\n")
            bitfile.write(bit)

s=Steganography('data/tweet.txt')
s.hiding()







