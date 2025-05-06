from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)



        

class MaxPool2x2(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        return X.reshape(B, C, H//2, 2, W//2, 2).max(axis=(3, 5))

    def backward(self, grad):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input)

        out_h, out_w = H // 2, W // 2
        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        region = self.input[b, c, i*2:i*2+2, j*2:j*2+2]
                        max_idx = np.unravel_index(np.argmax(region), (2, 2))
                        grad_input[b, c, i*2 + max_idx[0], j*2 + max_idx[1]] = grad[b, c, i, j]
        return grad_input



class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.train = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.train:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(np.float32)
            return X * self.mask / (1 - self.p)
        else:
            return X

    def backward(self, grad):
        return grad * self.mask / (1 - self.p) if self.train else grad



        

class Model_CNN(Layer):
    def __init__(self):
        self.layers = [
            conv2D(1, 8, 3, 1, 1, weight_decay=True, weight_decay_lambda=1e-4),
            ReLU(),
            MaxPool2x2(),
            Dropout(p=0.2),
            conv2D(8, 16, 3, 1, 1, weight_decay=True, weight_decay_lambda=1e-4),
            ReLU(),
            MaxPool2x2(),
            Dropout(p=0.3),
            conv2D(16, 32, 3, 1, 1, weight_decay=True, weight_decay_lambda=1e-4),
            ReLU(),
            Flatten(),
            Linear(32 * 7 * 7, 128, weight_decay=True, weight_decay_lambda=1e-4),
            ReLU(),
            Dropout(p=0.5),
            Linear(128, 10, weight_decay=True, weight_decay_lambda=1e-4)
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                param_list.append(layer.params)
            else:
                param_list.append(None)
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)
        for layer, param in zip(self.layers, param_list):
            if hasattr(layer, 'params') and param is not None:
                for key in param:
                    layer.params[key] = param[key]
                    setattr(layer, key, param[key])


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
