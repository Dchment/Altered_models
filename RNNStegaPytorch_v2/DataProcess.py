import gensim
import numpy as np
import torch
import multiprocessing
import os
import Configure
import collections


def data_read(file): #读取数据
    data=open(file,'r',encoding='utf-8')
    start_words = []
    sentences=[]
    for line in data:
        sentence=line.strip().split(" ") # strip移除字符串开头结尾
        start_word=sentence[0]
        start_words.append(start_word)
        sentences.append(["<BOS>"] + sentence + ["<EOS>"])   #加入语句开始符与结束符
    return start_words,sentences

def train_word2vec(sentences,embed_size,embed_dataset_name,model_savepath="./models/Word2Vec",w2vec_iter=5): #训练词向量模型
    embed_dataset_name += ".embed"  # 加入文件后缀名
    print("word2vec dataset name: "+embed_dataset_name)
    if(os.path.exists(os.path.join(model_savepath,embed_dataset_name))):
        print("Loading existing embeddings file")
        return gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_savepath,embed_dataset_name),binary=False)

    model = gensim.models.word2vec.Word2Vec(sg=0, workers=multiprocessing.cpu_count(),
                                                size=embed_size,min_count=0,iter=w2vec_iter)  # 构建模型
    model.build_vocab(sentences=sentences)  # 构建词典
    print("Training w2vec")
    print(model.corpus_count)
    model.train(sentences=sentences, total_words=model.corpus_count, epochs=model.iter)  # 训练模型

    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    model.wv.save_word2vec_format(os.path.join(model_savepath, embed_dataset_name))  # 存储训练完的Word2Vec模型内部参数，即词向量
    return gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_savepath, embed_dataset_name),binary=False)

def sentence_to_word2vec(sentence,w2v_model):#将一段语句转换为对应的词向量矩阵
    embedding_size=Configure.Config.embedding_size
    batch_size=Configure.Config.batch_size
    matrix=np.zeros((len(sentence),embedding_size))
    i=0
    for word in sentence:
        vec=np.array(w2v_model[word].reshape((1,embedding_size)))#调用词向量模型中储存的数据！
        matrix[i]=vec
        i+=1
    return matrix

class Dictionary(object):
    def __init__(self, sentences, vocab_drop):
        # sentences - array of sentences
        self._vocab_drop = vocab_drop
        if vocab_drop < 0:
            raise ValueError
        self._sentences = sentences
        self._word2idx = {}
        self._idx2word = {}
        self._words = []
        self.get_words()
        self._words.append("<unk>")
        self.build_vocabulary()
        self._mod_sentences()

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def sentences(self):
        return self._sentences

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def get_words(self):
        for line in self.sentences:
            for word in line:
                word = word if word in ["<EOS>", "<BOS>", "<PAD>", "<unk>"] else word.lower()
                self._words.append(word)

    def _mod_sentences(self):
        # for every sentence, if word not in vocab set to <unk>
        for i in range(len(self._sentences)):
            line = self._sentences[i]
            for j in range(len(line)):
                try:
                    self.word2idx[line[j]]
                except:
                    line[j]="<unk>"
            self._sentences[i] = line


    def build_vocabulary(self):
        counter = collections.Counter(self._words)#构建
        # words that occur less than 5 times don't include
        sorted_dict = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        sorted_dict = [(wd, count) for wd, count in sorted_dict
                       if count >= self._vocab_drop or wd in ["<unk>", "<BOS>", "<EOS>"]]
        # after sorting the dictionary, get ordered words
        words, _ = list(zip(*sorted_dict))#将排序后的字典“解压缩”变成单词和索引（zip*解压缩后返回一个对象）
        self._word2idx = dict(zip(words, range(1, len(words) + 1)))
        self._idx2word = dict(zip(range(1, len(words) + 1), words))
        # add <PAD> as zero
        self._idx2word[0] = "<PAD>"
        self._word2idx["<PAD>"] = 0

    def __len__(self):
        return len(self.idx2word)

def padding(data):
    max_len=max([len(line) for line in data])
    for line in data:
        if(len(line)<max_len):
            for i in range(len(line),max_len):
                line.append(0)
    return data

def prepare_data(data_raw, params):
    # get embeddings, prepare data
    print("building dictionary")
    data_dict = Dictionary(data_raw, params.vocab_drop)
    embed_arr = None

    w2_vec = train_word2vec(sentences=data_dict.sentences, embed_size=params.embedding_size,embed_dataset_name=Configure.Config.chosen_dataset, w2vec_iter=5)
    embed_arr = np.zeros([data_dict.vocab_size, params.embedding_size])
    for i in range(embed_arr.shape[0]):
        if i == 0:
            continue
        embed_arr[i] = w2_vec.word_vec(data_dict.idx2word[i])


    data = [[data_dict.word2idx[word] for word in sent[:-1]] for sent in data_dict.sentences
            if len(sent) < params.line_max_size - 2]
    labels = [[data_dict.word2idx[word] for word in sent[1:]] for sent in data_dict.sentences
              if len(sent) < params.line_max_size - 2]
    print("----Corpus_Information--- \n "
          "Raw data size: {} sentences \n Vocabulary size {}"
          "\n Limited data size {} sentences \n".format(len(data_raw), data_dict.vocab_size, len(data)))
    Configure.Config.vocab_size=data_dict.vocab_size

    return data, labels, embed_arr, data_dict

