from BPNN.NeuralNetwork import *
import numpy as np


b = np.array([[0, 0], [1, 1], [1, 0], [0, 1]]) #异或逻辑测试
c = np.array([[0], [0], [1], [1]])
a = BPnn(b, c, [2, 2, 1], 0.2)
a.train(10000)
a.predict(np.array([[0, 1]]))