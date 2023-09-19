class Layer:
    def __init__(self, input, output):
        self.input = input
        self.output = output
    
    def forward(self,input):
        raise NotImplementedError

    def backward(self,output_error,learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self,input_size,output_size):
        super().__init__(None,None)
        self.weight = np.random.rand(output_size,input_size) - 0.5
        self.bias = np.random.rand(output_size,1) - 0.5

    def forward(self,input):
        self.input = input
        self.output = self.weight.dot(input) + self.bias
        return self.output
    
    def backward(self,output_error,learning_rate):
        input_error = np.dot(self.weight.T,output_error)
        weight_error = np.dot(output_error,self.input.T)

        self.weight -= learning_rate*weight_error
        self.bias -= learning_rate*output_error

        # Error of next layer
        return input_error

class ActivationLayer(Layer):
    def __init__(self,activation, activation_derivative):
        super().__init__(None,None)
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward(self,input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward(self,output_error,learning_rate):
        # compute the input error for the activation fuction
        return output_error * self.activation_derivative(self.input)

class ReshapeLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return input.reshape(*self.output_shape)
    
    def backward(self,output_error,learning_rate):
        return output_error.reshape(*self.input_shape)