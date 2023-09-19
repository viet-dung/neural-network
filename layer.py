class Layer:
    def __init__(self, input, output):
        self.input = input
        self.output = output
    
    def forward(self,input):
        raise NotImplementedError

    def backward(self,output_error,learning_rate):
        raise NotImplementedError