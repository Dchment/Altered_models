import torch
import pandas
import numpy as np
import collections
import Configure
import os
class Data():
    def __init__(self):
        self.readfromcsv(Configure.Config.rawdata_path)
        self.sentences, self.start_words = self.read(Configure.Config.text_path)
        self.words=self.get_words(self.sentences)
        self.word2idx,self.idx2word=self.build_dictionary()

    def readfromcsv(self,path):
        f=open(path,'r',encoding='utf-8')
        if not os.path.exists(Configure.Config.text_path):
            text=open(Configure.Config.text_path,'w',encoding='utf-8')
        else:
            os.remove(Configure.Config.text_path)
            text = open(Configure.Config.text_path, 'w', encoding='utf-8')
        data = pandas.read_csv(f)
        data = data.values.tolist()
        for line in data:
            text.write(str(line[1]))
            text.write(' \n')
        f.close()



    def read(self,path):
        f = open(path, 'r', encoding='utf-8')
        start_words=[]
        sentences=[]
        for line in f:
           sentence=line.strip().split(" ")
           startword=sentence[0]
           sentences.append(sentence)
           start_words.append(startword)
        #print(sentences)
        #print(start_words)
        return sentences,start_words
    def get_words(self,sentences):
        words=[]
        for line in sentences:
            for word in line:
                words.append(word if word in ['\n',''] else word.lower())
        return words
    def build_dictionary(self):
        counter=collections.Counter(self.words)
        sorted_dict=sorted(counter.items(),key=lambda x: (-x[1], x[0]))
        sorted_dict= [(word, count) for word, count in sorted_dict
                       if count >= Configure.Config.vocab_drop or word in ["\n", ""]]
        words, counts = list(zip(*sorted_dict))
        word2idx = dict(zip(words, range(1, len(words) + 1)))
        idx2word = dict(zip(range(1, len(words) + 1), words))
        idx2word[0] = "<PAD>"
        word2idx["<PAD>"] = 0
        return word2idx,idx2word







