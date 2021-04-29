from tkinter import *
from tkinter import ttk
import sys
import run

S=run.Steganography()
window=Tk()
window.title("Steganography")
window.geometry('1024x768')

label_sm=Label(window,text='Enter your secret message:',anchor=W)
label_sm.place(x=20,y=20,width=200,height=20)
secret_m=Text(window)
secret_m.place(x=20,y=40,width=400,height=200)

encode_button=Button(window,text='隐写',command=lambda:S.encoder(secret_m.get("1.0","end")))
encode_button.place(x=20,y=240,width=40,height=20)

label_stego=Label(window,text='Stego text:',anchor=W)
label_stego.place(x=500,y=20,width=200,height=20)
stego_text=Text(window)
stego_text.place(x=500,y=40,width=400,height=200)

show_button=Button(window,text='显示',command=lambda:stego_text.insert("end",S.ste_message))
show_button.place(x=500,y=240,width=40,height=20)

label_st=Label(window,text='Enter your stego text:',anchor=W)
label_st.place(x=20,y=300,width=200,height=20)
ste=Text(window)
ste.place(x=20,y=320,width=400,height=200)

decode_button=Button(window,text='解码',command=lambda:S.decoder(ste.get("1.0","end")))
decode_button.place(x=20,y=520,width=40,height=20)

label_decode=Label(window,text='Secret message:',anchor=W)
label_decode.place(x=500,y=300,width=200,height=20)
decode_text=Text(window)
decode_text.place(x=500,y=320,width=400,height=200)

show_decode=Button(window,text='显示',command=lambda:decode_text.insert("end",S.decode_message))
show_decode.place(x=500,y=520,width=40,height=20)

label_mode=Label(window,text='Select your chosen mode:',anchor=W)
label_mode.place(x=20,y=560,width=200,height=20)

mode=ttk.Combobox(window,values=['arithmetic','huffman','bins'])
def go(*args):  #处理事件，*args表示可变参数
  S.mode=mode.get()
mode.bind("<<ComboboxSelected>>",go)
mode.place(x=20,y=580,width=200,height=20)
window.mainloop()

