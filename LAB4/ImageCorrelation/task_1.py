import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ImageCorrelation:
    def __init__(self, image_path):
        self.image = self.load_image(image_path)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def autocorrelation_rgb(self):
       # image = np.array(Image.open(image_path).convert('RGB'))
        image = np.copy(self.image)

        autocorrelation_sum = np.zeros_like(image[:, :, 0], dtype=np.float32)

        for c in range(3):  # c = 0 (R), 1 (G), 2 (B)
            channel = image[:, :, c]

            channel_fft = fft2(channel)
            autocorrelation_fft = ifft2(channel_fft * np.conj(channel_fft)) # умножение спектра Фурье на его комплексно-сопряжённое

            autocorrelation_sum += fftshift(np.abs(autocorrelation_fft))

        autocorrelation_sum -= autocorrelation_sum.min()
        autocorrelation_sum /= autocorrelation_sum.max()
        autocorrelation_sum *= 255

        return np.uint8(autocorrelation_sum)

    def autocorrelation_grayscale(self):
        image = np.array(Image.open(image_path).convert('L'))

        image_fft = fft2(image)
        autocorrelation_fft = ifft2(image_fft * np.conj(image_fft))

        autocorrelation = fftshift(np.abs(autocorrelation_fft))

        autocorrelation -= autocorrelation.min()
        autocorrelation /= autocorrelation.max()
        autocorrelation *= 255

        return np.uint8(autocorrelation)


if __name__ == "__main__":
    image_path = 'mario.png'

    correlation_calculator = ImageCorrelation(image_path)
    image = correlation_calculator.image

    autocorrelation_result = correlation_calculator.autocorrelation_rgb()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image, cmap='jet')
    axs[0].set_title("Image")
    axs[0].axis('off')

    axs[1].imshow(autocorrelation_result, cmap='jet')
    axs[1].set_title("Auto-Correlation Function")
    axs[1].axis('off')

    plt.savefig("autocorrelation_results_mario.png", dpi=300, bbox_inches='tight')
    plt.close()
