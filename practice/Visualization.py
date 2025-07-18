import matplotlib.pyplot as plt

class Visualization:
    def plotDigit(self, image_data):
        """
        Display a single digit image from the MNIST dataset.
        
        Args:
            image_data (numpy.ndarray): A 28x28 array representing a digit image
                                       with pixel values from 0-255
        
        This method:
        1. Reshapes the image data to 28x28 if needed
        2. Displays it using matplotlib with binary colormap (black/white)
        3. Removes axis labels for cleaner visualization
        """
        # Reshape to 28x28 if the image is flattened (784 elements)
        if image_data.shape == (784,):
            image = image_data.reshape(28, 28)
        else:
            image = image_data
            
        # Create the plot
        plt.imshow(image, cmap="binary")
        plt.axis("off")  # Remove axis labels and ticks
        plt.title("MNIST Digit")
        plt.show()


