import numpy as np
import bitarray
import sys
import re
import math

from utils import get_model, encode_context

from arithmetic import encode_arithmetic, decode_arithmetic
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from sample import sample

# encoder函数编码信息形成隐写文本，decoder函数解码隐写文本
class Steganography():
    def __init__(self):
        self.enc, self.model = get_model(model_name='gpt2')
        ## PARAMETERS
        self.message_str = ""

        self.unicode_enc = False
        self.mode = 'arithmetic'# 可选择'arithmetic','huffman','bins'三种编码方式，或使用'sample'不隐写直接生成文本
        self.block_size = 3  # for huffman and bins
        self.temp = 0.9  # for arithmetic
        self.precision = 26  # for arithmetic
        self.sample_tokens = 100  # for sample
        self.topk = 300  # topk机制，用于限制词汇选择数
        self.finish_sent = False  # whether or not to force finish sent. If so, stats displayed will be for non-finished sentence
        self.ste_message=""
        self.decode_message=""
    def encoder(self,message):
        ## VALIDATE PARAMETERS
        if self.mode not in ['arithmetic', 'huffman', 'bins', 'sample']:
            raise NotImplementedError

        if self.mode == 'bins':
            bin2words, words2bin = get_bins(len(self.enc.encoder), self.block_size)

        context = \
"""Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.


"""
        self.message_str=message
        context_tokens = encode_context(context, self.enc)

        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------

        # First encode message to uniform bits, without any context
        # (not essential this is arithmetic vs ascii, but it's more efficient when the message is natural language)
        if self.unicode_enc:
            ba = bitarray.bitarray()
            ba.frombytes(self.message_str.encode('utf-8'))
            message = ba.tolist()
        else:
            message_ctx = [self.enc.encoder['<|endoftext|>']]
            self.message_str += '<eos>'
            message = decode_arithmetic(self.model, self.enc, self.message_str, message_ctx, precision=40, topk=60000)

        # Next encode bits into cover text, using arbitrary context
        Hq = 0
        print(self.mode)
        if self.mode == 'arithmetic':
            out, nll, kl, words_per_bit, Hq = encode_arithmetic(self.model, self.enc, message, context_tokens, temp=self.temp,
                                                                finish_sent=self.finish_sent, precision=self.precision, topk=self.topk)
        elif self.mode == 'huffman':
            out, nll, kl, words_per_bit = encode_huffman(self.model, self.enc, message, context_tokens, self.block_size,
                                                         finish_sent=self.finish_sent)
        elif self.mode == 'bins':
            out, nll, kl, words_per_bit = encode_block(self.model, self.enc, message, context_tokens, self.block_size, bin2words,
                                                       words2bin,
                                                       finish_sent=self.finish_sent)
        elif self.mode == 'sample':
            out, nll, kl, Hq = sample(self.model, self.enc, self.sample_tokens, context_tokens, temperature=self.temp, topk=self.topk)
            words_per_bit = 1
        text = self.enc.decode(out)

        print(message)
        print(len(message))
        print("=" * 40 + " Encoding " + "=" * 40)
        print(text)
        print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (
            math.exp(nll), kl, words_per_bit, 1 / words_per_bit, Hq / 0.69315))
        self.ste_message=text

    def decoder(self,text):
        ## VALIDATE PARAMETERS
        if self.mode not in ['arithmetic', 'huffman', 'bins', 'sample']:
            raise NotImplementedError

        if self.mode == 'bins':
            bin2words, words2bin = get_bins(len(self.enc.encoder), self.block_size)


        context = \
"""Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.


"""

        if self.unicode_enc:
            ba = bitarray.bitarray()
            ba.frombytes(text.encode('utf-8'))
            message = ba.tolist()
        else:
            message_ctx = [self.enc.encoder['<|endoftext|>']]
            self.message_str += '<eos>'
            message = decode_arithmetic(self.model, self.enc, text, message_ctx, precision=40, topk=60000)

        context_tokens = encode_context(context, self.enc)
        # Decode binary message from bits using the same arbitrary context
        if self.mode != 'sample':
            if self.mode == 'arithmetic':
                message_rec = decode_arithmetic(self.model, self.enc, text, context_tokens, temp=self.temp, precision=self.precision,
                                                topk=self.topk)
            elif self.mode == 'huffman':
                message_rec = decode_huffman(self.model, self.enc, text, context_tokens, self.block_size)
            elif self.mode == 'bins':
                message_rec = decode_block(self.model, self.enc, text, context_tokens, self.block_size, bin2words, words2bin)

            print("=" * 40 + " Recovered Message " + "=" * 40)
            print(message_rec)
            print("=" * 80)
            # Finally map message bits back to original text
            if self.unicode_enc:
                message_rec = [bool(item) for item in message_rec]
                ba = bitarray.bitarray(message_rec)
                reconst = ba.tobytes().decode('utf-8', 'ignore')
            else:
                reconst = encode_arithmetic(self.model, self.enc, message_rec, message_ctx, precision=40, topk=60000)
                reconst = self.enc.decode(reconst[0])
            print(reconst)
            self.decode_message=reconst
'''
s=Steganography()
text=s.encode()
s.decode(text)
'''