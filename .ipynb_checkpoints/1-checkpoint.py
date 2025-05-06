
# test_train.py
# Example: Train the model. The runner is implemented, while the model used for training need your implementation.

import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
# from augmentation_utils import random_shift  # Optional if data augmentation used

# fixed seed for reproducibility
np.random.seed(309)

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# Shuffle and split
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# Normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

# Optional data augmentation
# train_imgs_rs = train_imgs.reshape(-1, 1, 28, 28)
# train_imgs_rs = random_shift(train_imgs_rs, shift_range=2)
# train_imgs = train_imgs_rs.reshape(-1, 28*28)

# Model, optimizer, loss, scheduler
linear_model = nn.models.Model_MLP([784, 512, 128, 10], 'ReLU', [1e-4]*3)
optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=linear_model, mu=0.9)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

# Train
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=5, log_iters=100, save_dir='./best_models')

# Plot training curve
_, axes = plt.subplots(1, 2, figsize=(10, 4))
plot(runner, axes)
plt.savefig("training_curve.png")
plt.show()
