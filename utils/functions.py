import numpy as np
import torch
import matplotlib.pyplot as plt

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = 1)
    x = np.expand_dims(x, axis=1)
    print("x shape=", x.shape)
    #x = torch.argmax(image, axis = 1)
    return x


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    fig = plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()

    return fig