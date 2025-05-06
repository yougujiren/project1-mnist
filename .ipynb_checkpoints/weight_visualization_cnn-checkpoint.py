import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN()
model.load_model(r'.\cnn_models\best_model.pickle')

conv_weights = None
for layer in model.layers:
    if hasattr(layer, 'params') and 'W' in layer.params:
        if len(layer.params['W'].shape) == 4:  # [out_channels, in_channels, H, W]
            conv_weights = layer.params['W']
            break

if conv_weights is not None:
    out_channels = conv_weights.shape[0]
    fig, axes = plt.subplots(1, out_channels, figsize=(2 * out_channels, 2))
    for i in range(out_channels):
        kernel = conv_weights[i, 0]  
        axes[i].imshow(kernel, cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Convolution Kernels')
    plt.tight_layout()
    plt.show()
else:
    print("No conv2D layer found.")