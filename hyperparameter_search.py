import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle
import time


def load_data():
    with gzip.open(r'.\dataset\MNIST\train-images-idx3-ubyte.gz', 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    with gzip.open(r'.\dataset\MNIST\train-labels-idx1-ubyte.gz', 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    idx = np.random.permutation(np.arange(num))
    train_imgs = train_imgs[idx] / 255.
    train_labs = train_labs[idx]
    return train_imgs[10000:], train_labs[10000:], train_imgs[:10000], train_labs[:10000]


train_imgs, train_labs, valid_imgs, valid_labs = load_data()
train_imgs = train_imgs[:5000]
train_labs = train_labs[:5000]


hidden_configs = [[128, 10], [256, 128, 10], [512, 128, 10]]
lrs = [0.01, 0.03]
decays = [1e-4, 5e-4]

results = []
start_time = time.time()

for hidden in hidden_configs:
    for lr in lrs:
        for decay in decays:
            print(f"\n>>> Testing config: hidden={hidden}, lr={lr}, decay={decay}")

            model = nn.models.Model_MLP([784] + hidden, 'ReLU', [decay] * len(hidden))
            optimizer = nn.optimizer.MomentGD(init_lr=lr, model=model, mu=0.9)
            loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

            runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64)
            runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                         num_epochs=5, log_iters=100)  # epoch 从2增加到5

            score, loss = runner.evaluate([valid_imgs, valid_labs])
            results.append(((hidden, lr, decay), score))
            print(f"Validation Accuracy = {score:.4f} | Loss = {loss:.4f}")

end_time = time.time()
print(f"\nTotal search time: {(end_time - start_time):.2f} seconds")

results.sort(key=lambda x: x[1], reverse=True)
print("\nTop configurations:")
for cfg, acc in results:
    print(f"Hidden={cfg[0]}, LR={cfg[1]}, Decay={cfg[2]} => Acc={acc:.4f}")
