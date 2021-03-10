import torch
import utils
from gpt2_sentence_gen_mod import Steganography

if __name__ == '__main__':
    test=Steganography()
    text=test.encode()
    print(test.text_bit)
    message=test.decode(text)
    print(message)
    print(len(message))