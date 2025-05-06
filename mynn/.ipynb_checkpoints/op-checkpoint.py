from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

def kaiming_init(size):
    fan_in = np.prod(size[1:])
    return np.random.randn(*size) * np.sqrt(2. / fan_in)


class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.grads = {'W': None, 'b': None}

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad):
        self.grads['W'] = np.dot(self.input.T, grad) / grad.shape[0]
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / grad.shape[0]
        return np.dot(grad, self.W.T)

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}




class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=lambda size: np.random.randn(*size) * np.sqrt(2. / np.prod(size[1:]))
, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels, 1))  
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        F, _, HH, WW = self.W.shape
        out_h = (H - HH + 2 * self.padding) // self.stride + 1
        out_w = (W - WW + 2 * self.padding) // self.stride + 1
        out = np.zeros((B, F, out_h, out_w))
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        for i in range(out_h):
            for j in range(out_w):
                x_slice = X_padded[:, :, i*self.stride:i*self.stride+HH, j*self.stride:j*self.stride+WW]
                for k in range(F):
                    out[:, k, i, j] = np.sum(x_slice * self.W[k, :, :, :], axis=(1, 2, 3)) + self.b[k, 0]
        return out

    def backward(self, grad):
        B, C, H, W = self.input.shape
        F, _, HH, WW = self.W.shape
        _, _, out_h, out_w = grad.shape
        X_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        dX = np.zeros_like(X_padded)
        dW = np.zeros_like(self.W)
        db = np.zeros((F, 1))

        for i in range(out_h):
            for j in range(out_w):
                x_slice = X_padded[:, :, i*self.stride:i*self.stride+HH, j*self.stride:j*self.stride+WW]
                for k in range(F):
                    dW[k] += np.sum(x_slice * grad[:, k:k+1, i:i+1, j:j+1], axis=0)
                for n in range(B):
                    for k in range(F):
                        dX[n, :, i*self.stride:i*self.stride+HH, j*self.stride:j*self.stride+WW] += grad[n, k, i, j] * self.W[k]
        db = np.sum(grad, axis=(0, 2, 3)).reshape(F, 1)
        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]
        self.grads['W'] = dW / B
        self.grads['b'] = db / B
        return dX




        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output




class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10):
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.labels = labels
        self.preds = softmax(predicts)
        batch_size = predicts.shape[0]
        self.grads = self.preds.copy()
        self.grads[np.arange(batch_size), labels] -= 1
        self.grads /= batch_size
        loss = -np.log(self.preds[np.arange(batch_size), labels] + 1e-12)
        return np.mean(loss)

    def backward(self):
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition