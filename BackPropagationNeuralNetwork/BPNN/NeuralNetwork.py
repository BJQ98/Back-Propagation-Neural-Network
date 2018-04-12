import numpy as np
# b = np.random.rand(3, 2)
# print(b)
# d = np.array([[1]]*3).T


def sigmoid(x):     #激活函数
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):      #激活函数导数
    return x*(1-x)


class BPnn:                                      #例如[2,3,4]类型的各层数和每层神经元数量
    def __init__(self, input_matrix, output_matrix, layer_size, learn_rate):
        self.layers = []        #每一层的值
        self.input = input_matrix
        #print(type(self.input))
        self.output = output_matrix
        self.layer_neural_count = len(layer_size) - 1   #神经层数
        self.layer_size = layer_size
        self.weight = {}    #权重值
        self.bias = {}      #偏置值
        self.learn_rate = learn_rate    #学习率
        for i in range(1, self.layer_neural_count+1):
            self.weight[i-1] = np.random.rand(self.layer_size[i-1], self.layer_size[i])
            self.bias[i-1] = np.random.rand(1, self.layer_size[i])

    def forward(self, _input):      #向前传播求各层的输出值
        self.layers = []    #每次需要把已保存的值清空
        #print(_input)
        self.layers.append(_input)      #已给输入加进保存的值中
        for i in range(self.layer_neural_count):
            raw_value = np.dot(self.layers[i], self.weight[i]) + self.bias[i]   #求各层各神经元的未激活值
            active_value = sigmoid(raw_value)   #用激活函数进行非线性变换
            self.layers.append(active_value)    #将每层求得的值加入栈空间保存，实现每次新求的值利用上一层的数值

    def back_propagation(self, _output):
        update_list = []
        theta_output = (self.layers[-1] - _output) * sigmoid_derivative(self.layers[-1])   #用误差平方和的一半对计算输出求导
        update_list.append(theta_output)
        for i in range(self.layer_neural_count-1, 0, -1):                                  #用公式对后向传播过程进行计算
            theta_hidden = np.dot(update_list[-1], self.weight[i].T) * sigmoid_derivative(self.layers[i])
            update_list.append(theta_hidden)
        update_list.reverse()
        change_weight = {}
        change_bias = {}

        for i in range(len(update_list)):
            change_weight[i] = np.dot(self.layers[i].T, update_list[i]) * self.learn_rate
            change_bias[i] = update_list[i] * self.learn_rate

        for i in range(len(update_list)):       #更新权值和偏置值
            self.weight[i] -= change_weight[i]
            self.bias[i] -= change_bias[i]

        return (sum((self.layers[-1]-_output)**2)/2)

    def train(self, count):
        j = 0          #迭代次数
        for i in range(count):
            for x, y in zip(self.input, self.output):
                j += 1
                self.forward(x.reshape(1, len(x)))      #解决迭代对象矩阵形状丢失问题
                if self.back_propagation(y.reshape(1, len(y))) < 0.001: #误差小于0.001直接完成训练
                    print(j)
                    return

    def predict(self, input_array):
        self.layers = []  # 每次需要把已保存的值清空
        self.layers.append(input_array)  # 已给输入加进保存的值中
        for i in range(self.layer_neural_count):
            raw_value = np.dot(self.layers[i], self.weight[i]) + self.bias[i]  # 求各层各神经元的未激活值
            active_value = sigmoid(raw_value)  # 用激活函数进行非线性变换
            self.layers.append(active_value)
        print(self.layers)

