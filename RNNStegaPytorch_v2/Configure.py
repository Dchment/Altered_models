class Config(object):
    learning_rate=0.001#学习率
    num_layers=2#层数
    num_steps=50#步数
    batch_size=32#批次大小
    hidden_size=800#隐藏层神经元总数
    iteration=1#迭代次数
    keep_prob=0.5#dropout层保留神经元率
    vocab_size=46088#词汇表总量
    embedding_size=300#嵌入层词向量维度
    model_path='./models/'
    len_of_generation=40#生成句子最大长度
    chosen_dataset='tweet'
    line_max_size = 100#生成句子最多条数
    vocab_drop=5#词频阈值（低于其会被忽略）


