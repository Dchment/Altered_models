import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
import Configure

class decoder(nn.Module):
    def __init__(self,Config):
        super(decoder, self).__init__()
        self.batch_size = Config.batch_size
        self.hidden_size = Config.hidden_size
        self.vocab_size = Config.vocab_size
        self.embed_size = Config.embed_size
        self.bidirectional = Config.bidirectional
        self.dropout = Config.dropout
        self.seq_length=Config.seq_length
        self.layer_size=Config.layer_size
        self.output_size=Config.output_size
        self.embedding=nn.Embedding(self.vocab_size,self.embed_size)
        self.lstm=nn.LSTM(input_size=self.embed_size+self.hidden_size,hidden_size=self.hidden_size,num_layers=self.layer_size,dropout=self.dropout)
        self.wp=nn.Linear(self.hidden_size,self.vocab_size)

    def forward(self, input_sentences,encoder_outputs,batch_size=None):
        self.h_0 = torch.autograd.Variable(
            torch.zeros(self.layer_size , input_sentences.shape[0], self.hidden_size))
        self.c_0 = torch.autograd.Variable(
            torch.zeros(self.layer_size , input_sentences.shape[0], self.hidden_size))
        output1 = self.embedding(input_sentences)
        output1 = output1.permute(1, 0, 2)
        print(output1.shape)
        a=[]
        for i in range(output1.shape[0]):
            a.append(torch.cat((output1[i,:,:] ,encoder_outputs),dim=1))
        inputs=torch.cat(a).reshape([-1,input_sentences.shape[0],self.embed_size+self.hidden_size])
        print(inputs.shape)
        lstm_output, (h_final, c_final) = self.lstm(inputs, (self.h_0, self.c_0))
        print(lstm_output.shape)
        distribute=nn.functional.softmax(self.wp(lstm_output),dim=2)
        print(distribute.shape)


        return distribute