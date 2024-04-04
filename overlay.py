from torchquantum.dataset import NoisyMNISTDataset
import matplotlib.pyplot as plt

images = []
digits = [1,5,9]
for i in range(10):
    mnist = NoisyMNISTDataset(
        root="mnist_data",
        split="train",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=28,
        resize=28,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=tuple(range(10)),
        n_test_samples=100,
        n_valid_samples=1000,
        fashion=False,
        n_train_samples=10000,
        std_dev=0.5*i,
    )
    
    
    for digit in digits:
        item = None
        ind = 0
        for item in mnist:
            if item['digit'] == digit: break
        image_tensor = item['image'].squeeze()
        images.append(image_tensor)


# Define the dimensions of the grid
num_rows = 3
num_cols = 10
ind = 0
# Create a figure and axis
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6))

# Iterate over the axes and images
for j in range(num_cols):
    for i in range(num_rows):
        img = images[ind]
        
        # Display the image
        axes[i, j].imshow(img, cmap='gray')

        axes[i,j].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

        axes[i,j].tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,
        labelleft=False)
        
        #axes[i, j].axis('off')

        ind += 1

# Adjust layout
plt.subplots_adjust(wspace=0.01, hspace=0.01)

for j in range(num_cols):
    axes[num_rows-1, j].set_xlabel(str(round(0.5*j,3)), fontsize=14, labelpad=10)

# Show the grid of images
plt.suptitle("MNIST Digits with Increasing Degrees of Noise Added")
# plt.ylabel("Digit")
# plt.xlabel("Standard Deviation of Gaussian Noise")
plt.figtext(.5,.95,'Foo Bar', fontsize=18, ha='center')
plt.figtext(.5,.9,"Standard Deviation of Gaussian Noise",fontsize=18,ha='center')
plt.savefig('overlay.eps', format='eps')
plt.savefig('overlay.png')
