import matplotlib.pyplot as plt
import os

# Directory containing the combined images
directory = r'C:\Users\aryes\Desktop\YEAR 2\mrm\datasets\DID-dataset\DID-MDN-test'

# Create a directory to store the separated images
output_directory = r'C:\Users\aryes\Desktop\YEAR 2\mrm\GANproject\dataset\test_data'

# Loop through each image file
for i in range(20):
    # Load the combined image
    combined_image_path = os.path.join(directory, f'{i}.jpg')
    combined_image = plt.imread(combined_image_path)
    
    # Assuming the combined image contains the noisy image on the left and ground truth on the right
    height, width, _ = combined_image.shape

    # Split the image into two parts
    noisy_image = combined_image[:, :width//2]
    ground_truth = combined_image[:, width//2:]

    # Save the separated images
    plt.imsave(os.path.join(output_directory, f'noisy_{i}.jpg'), noisy_image)
    plt.imsave(os.path.join(output_directory, f'ground_truth_{i}.jpg'), ground_truth)
