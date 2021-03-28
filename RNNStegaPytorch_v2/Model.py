import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class model(nn.Module):
    def __init__(self,config):
        super(model, self).__init__()
        self.hidden_size=config.hidden_size
        self.num_steps=num_steps=config.num_steps
        self.num_layers=config.num_layers
        self.size=config.hidden_size
        self.vocab_size=config.vocab_size
        self.lr=config.learning_rate
        self.keep_prob=config.keep_prob
        self.embedding_size=config.embedding_size
        self.embedding=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_size)
        self.lstm=nn.LSTM(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.keep_prob,batch_first=True)
        self.dropout=nn.Dropout(self.keep_prob)
        self.weight_prediction=nn.Linear(self.hidden_size,self.vocab_size)
        self.softmax=nn.LogSoftmax(dim=2)


    def forward(self,inputs):
        output1=self.embedding(inputs)
        #print(output1.shape)
        output2,(h_0,c_0)=self.lstm(output1.transpose(0,1))
        #print(output2.shape)
        output3=self.dropout(output2)
        #print(output3.shape)
        output4=self.weight_prediction(output3)
        #print(output4.shape)
        output=self.softmax(output4)
        return output


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self,output):
        loss=0
        for i in range(output.shape[0]-1):
            for j in range(output.shape[2]-1):
                loss+=torch.log(output[i][0][j])
        loss=-loss
        return loss


