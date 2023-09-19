class Network:
    def __init__(self,loss_fn,loss_fn_derivative):
        self.layers = []
        self.loss = None
        self.loss_fn = loss_fn
        self.loss_fn_derivative = loss_fn_derivative
    
    def add(self,layer):
        self.layers.append(layer)
    
    def fit(self, data, labels, epochs, learning_rate):
        for i in range(epochs):
            loss = 0
            for j in range(len(data)):
                output = data[j]
                label = labels[j]
                for layer in self.layers:
                    output = layer.forward(output)

                loss += self.loss_fn(output,label)
                
                output_error = self.loss_fn_derivative(output,label)
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error,learning_rate)
            
            loss /= len(data)
            print("Epoch %i  Error: %f" %(i,loss))
        
    def predict(self,data,labels):
            loss = 0
            for j in range(len(data)):
                output = data[j]
                label = labels[j]
                for layer in self.layers:
                    output = layer.forward(output)

                loss += self.loss_fn(output,label)
                
            
            loss /= len(data)
            print("Error: %f" %(loss))