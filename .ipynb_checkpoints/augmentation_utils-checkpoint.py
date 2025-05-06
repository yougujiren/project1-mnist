
import numpy as np

def random_shift(images, shift_range=2, direction="both"):
    
    B, C, H, W = images.shape
    shifted_images = np.zeros_like(images)

    for i in range(B):
        img = images[i, 0]
        dx = np.random.randint(-shift_range, shift_range + 1) if direction in ["both", "horizontal"] else 0
        dy = np.random.randint(-shift_range, shift_range + 1) if direction in ["both", "vertical"] else 0

        shifted = np.zeros((H, W))
        src_x_start = max(0, dx)
        src_y_start = max(0, dy)
        dst_x_start = max(0, -dx)
        dst_y_start = max(0, -dy)

        copy_h = H - abs(dy)
        copy_w = W - abs(dx)

        shifted[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = \
            img[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

        shifted_images[i, 0] = shifted

    return shifted_images

