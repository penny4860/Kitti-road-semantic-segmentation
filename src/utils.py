
import matplotlib.pyplot as plt

def plot_img(images, show=True, save_filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=len(images))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap="gray")
    if show:
        plt.show()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        print("{} is saved".format(save_filename))
