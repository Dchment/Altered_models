import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
import Configure
class encoder(nn.Module):
    def __init__(self,Config):
        super(encoder, self).__init__()
        self.batch_size = Config.batch_size
        self.hidden_size = Config.hidden_size
        self.vocab_size = Config.vocab_size
        self.embed_size = Config.embed_size
        self.bidirectional = Config.bidirectional
        self.dropout = Config.dropout
        self.seq_length=Config.seq_length
        self.layer_size=Config.layer_size
        self.output_size=Config.output_size
        self.attention_size=Config.attention_size
        self.embedding=nn.Embedding(self.vocab_size,self.embed_size)

        self.lstm=nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.layer_size,dropout=self.dropout,
                          bidirectional=self.bidirectional)


        '''
        self.w_wp=torch.autograd.Variable(torch.zeros(self.hidden_size,self.vocab_size))
        self.b_wp=torch.autograd.Variable(torch.zeros(self.vocab_size,))
        '''

        self.w_omega = torch.autograd.Variable(torch.zeros(self.hidden_size * (2 if self.bidirectional else 1), self.attention_size))
        self.u_omega = torch.autograd.Variable(torch.zeros(self.attention_size))


        '''
        attention layer:
        M=tanh(H)
        a=softmax(W^T*M)
        c=H*a^T
        '''
    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*(2 if self.bidirectional else 1))
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*(2 if self.bidirectional else 1)])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*(2 if self.bidirectional else 1))
        print(output_reshape.size())
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        print(attn_tanh.size())
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
        print(attn_hidden_layer.size())
        alphas = torch.Tensor.reshape(torch.nn.functional.softmax(attn_hidden_layer, dim=0), [-1, self.seq_length])
        '''
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)
        '''
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.seq_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(alphas_reshape.size())
        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(state.size())
        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        print(attn_output.size())
        return attn_output

    def forward(self, input_sentences, batch_size=None):
        output1 = self.embedding(input_sentences)
        output1 = output1.permute(1, 0, 2)
        print(output1.shape)
        h_0 = torch.autograd.Variable(torch.zeros(self.layer_size*(2 if self.bidirectional else 1), input_sentences.shape[0], self.hidden_size))
        c_0 = torch.autograd.Variable(torch.zeros(self.layer_size*(2 if self.bidirectional else 1), input_sentences.shape[0], self.hidden_size))
        lstm_output, (h_final, c_final) = self.lstm(output1, (h_0, c_0))
        print(lstm_output.shape)
        attn_output = self.attention_net(lstm_output)

        return attn_output,(h_final, c_final)

