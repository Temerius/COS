import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage import io
import argparse

def match_image(image_path, fragment_path):
    image = io.imread(image_path)
    fragment = io.imread(fragment_path)

    result = match_template(image, fragment, True)

    ij = np.unravel_index(np.argmax(result), np.array(result).shape)
    y, x, z = ij


    fig = plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(fragment, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('Fragment')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('Image')
    # Выделяем найденную область на изображении
    h, w = fragment.shape[:2]
    top_left_x = x - w // 2
    top_left_y = y - h // 2
    rect = plt.Rectangle((top_left_x, top_left_y), w, h, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('Correlation image')
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.tight_layout()
    plt.savefig('correlation_results_mario.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    match_image("mario.png", "coin.png")
