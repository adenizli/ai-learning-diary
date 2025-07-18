from Database import Database
from Visualization import Visualization

# Create database instance
database = Database()

# Create visualization instance
visualization = Visualization()

# Load the MNIST dataset from local files
database.initMNIST()

# Display some basic information about the loaded data
print(f"\n" + "="*50)
print("MNIST Dataset Information:")
print("="*50)
print(f"Training images: {database.trainSet.shape}")
print(f"Training labels: {database.trainLabel.shape}")
print(f"Test images: {database.testSet.shape}")
print(f"Test labels: {database.testLabel.shape}")

# Show the first training image and its label
print(f"\nFirst training image label: {database.trainLabel[0]}")
print("Displaying the first training image...")

# Plot the first digit
visualization.plotDigit(database.trainSet[0])