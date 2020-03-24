"""
Author: Zhou Chen
Date: 2020/3/24
Desc: desc
"""
import matplotlib.pyplot as plt
import pickle
plt.style.use('fivethirtyeight')

f = open('his.pkl', 'rb')
his = pickle.load(f)
plt.figure(figsize=(16, 8))
plt.plot(list(range(len(his['loss']))), his['loss'], label="loss", c='b')
plt.plot(list(range(len(his['val_loss']))), his['val_loss'], label="val_loss", c='r')
plt.savefig('his.png')
plt.show()