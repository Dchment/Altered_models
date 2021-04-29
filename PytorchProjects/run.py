import torch
import Configure
import Encoder
import Decoder
import data_process
c=Configure.Config()
encode_model= Encoder.encoder(c)
a=torch.randint(low=0,high=1000,size=(c.batch_size,c.seq_length)).long()
print(a)
output,hidden=encode_model(a)


decode_model=Decoder.decoder(c)
b=torch.randint(low=0,high=1000,size=(c.batch_size,c.seq_length)).long()
print(b)
output2=decode_model(b,output)
print(output2)
