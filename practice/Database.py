import numpy as np
import struct
import os

class Database:
    def initMNIST(self):
        """
        Initialize MNIST dataset by loading from local IDX format files.
        This method replaces the fetch_openml approach and reads directly from
        the downloaded MNIST files in the data/ directory.
        
        The method creates four main data fields:
        - trainSet: Training images (60,000 images of 28x28 pixels)
        - trainLabel: Training labels (60,000 labels, 0-9 digits)
        - testSet: Test images (10,000 images of 28x28 pixels)  
        - testLabel: Test labels (10,000 labels, 0-9 digits)
        """
        
        # Define the paths to our local MNIST data files
        # These files are in IDX format - a simple binary format for machine learning data
        # Use os.path.dirname(__file__) to get the directory where this Database.py file is located
        # This ensures the path works regardless of where the script is run from
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
        train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
        test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
        test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")
        
        # Load training data
        print("Loading training images...")
        self.trainSet = self._load_images(train_images_path)
        print("Loading training labels...")
        self.trainLabel = self._load_labels(train_labels_path)
        
        # Load test data
        print("Loading test images...")
        self.testSet = self._load_images(test_images_path)
        print("Loading test labels...")
        self.testLabel = self._load_labels(test_labels_path)
        
        # Print summary information about the loaded dataset
        print(f"\nDataset loaded successfully!")
        print(f"Training set: {self.trainSet.shape[0]} images of size {self.trainSet.shape[1]}x{self.trainSet.shape[2]}")
        print(f"Training labels: {self.trainLabel.shape[0]} labels")
        print(f"Test set: {self.testSet.shape[0]} images of size {self.testSet.shape[1]}x{self.testSet.shape[2]}")
        print(f"Test labels: {self.testLabel.shape[0]} labels")
        
    def _load_images(self, filepath):
        """
        Load images from an IDX3 format file (3D array: number of images, height, width).
        
        IDX format structure for images:
        - 4 bytes: magic number (2051 for images)
        - 4 bytes: number of images
        - 4 bytes: number of rows (height) 
        - 4 bytes: number of columns (width)
        - pixel data: unsigned bytes (0-255) representing grayscale values
        
        Args:
            filepath (str): Path to the IDX3 image file
            
        Returns:
            numpy.ndarray: 3D array of images with shape (num_images, height, width)
        """
        with open(filepath, 'rb') as f:
            # Read the header information (16 bytes total)
            # '>I' means big-endian unsigned integer (4 bytes)
            magic = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            height = struct.unpack('>I', f.read(4))[0]
            width = struct.unpack('>I', f.read(4))[0]
            
            # Verify this is indeed an image file (magic number should be 2051)
            if magic != 2051:
                raise ValueError(f'Invalid magic number {magic}, expected 2051 for image file')
            
            # Read all the pixel data at once
            # Each pixel is 1 byte (unsigned integer 0-255)
            pixel_data = f.read(num_images * height * width)
            
            # Convert the raw bytes to a numpy array
            # dtype=np.uint8 means unsigned 8-bit integers (0-255)
            images = np.frombuffer(pixel_data, dtype=np.uint8)
            
            # Reshape the flat array into the proper 3D structure
            # -1 means "figure out this dimension automatically"
            images = images.reshape(num_images, height, width)
            
            return images
    
    def _load_labels(self, filepath):
        """
        Load labels from an IDX1 format file (1D array: number of labels).
        
        IDX format structure for labels:
        - 4 bytes: magic number (2049 for labels)
        - 4 bytes: number of labels
        - label data: unsigned bytes (0-9) representing digit classes
        
        Args:
            filepath (str): Path to the IDX1 label file
            
        Returns:
            numpy.ndarray: 1D array of labels with shape (num_labels,)
        """
        with open(filepath, 'rb') as f:
            # Read the header information (8 bytes total)
            magic = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            
            # Verify this is indeed a label file (magic number should be 2049)
            if magic != 2049:
                raise ValueError(f'Invalid magic number {magic}, expected 2049 for label file')
            
            # Read all the label data at once
            # Each label is 1 byte (0-9 for digits)
            label_data = f.read(num_labels)
            
            # Convert the raw bytes to a numpy array
            labels = np.frombuffer(label_data, dtype=np.uint8)
            
            return labels
    
    def get_sample_image(self, dataset='train', index=0):
        """
        Helper method to get a single image for visualization or testing.
        
        Args:
            dataset (str): Either 'train' or 'test' to specify which dataset
            index (int): Index of the image to retrieve
            
        Returns:
            tuple: (image, label) where image is 2D numpy array and label is integer
        """
        if dataset == 'train':
            if index >= len(self.trainSet):
                raise ValueError(f"Index {index} is out of range for training set (max: {len(self.trainSet)-1})")
            return self.trainSet[index], self.trainLabel[index]
        elif dataset == 'test':
            if index >= len(self.testSet):
                raise ValueError(f"Index {index} is out of range for test set (max: {len(self.testSet)-1})")
            return self.testSet[index], self.testLabel[index]
        else:
            raise ValueError("Dataset must be either 'train' or 'test'")
    
    def get_flattened_data(self, dataset='train'):
        """
        Get flattened image data suitable for traditional ML classifiers.
        Converts 3D image arrays (samples, height, width) to 2D (samples, features).
        
        Args:
            dataset (str): Either 'train' or 'test' to specify which dataset
            
        Returns:
            numpy.ndarray: 2D array with shape (num_samples, height*width)
        """
        if dataset == 'train':
            return self.trainSet.reshape(self.trainSet.shape[0], -1)
        elif dataset == 'test':
            return self.testSet.reshape(self.testSet.shape[0], -1)
        else:
            raise ValueError("Dataset must be either 'train' or 'test'")
    
    def get_binary_classification_targets(self, target_digit, dataset='train'):
        """
        Create binary classification targets for detecting a specific digit.
        
        Args:
            target_digit (int): The digit to detect (0-9)
            dataset (str): Either 'train' or 'test' to specify which dataset
            
        Returns:
            numpy.ndarray: Boolean array where True indicates the target digit
        """
        if not (0 <= target_digit <= 9):
            raise ValueError("Target digit must be between 0 and 9")
            
        if dataset == 'train':
            targets = self.trainLabel == target_digit
            print(f"Created binary targets for digit {target_digit}: {targets.sum()} positive examples out of {len(targets)}")
            return targets
        elif dataset == 'test':
            targets = self.testLabel == target_digit
            print(f"Test set binary targets for digit {target_digit}: {targets.sum()} positive examples out of {len(targets)}")
            return targets
        else:
            raise ValueError("Dataset must be either 'train' or 'test'")
    
    def get_single_flattened_image(self, dataset='train', index=0):
        """
        Get a single image flattened and ready for prediction.
        
        Args:
            dataset (str): Either 'train' or 'test' to specify which dataset
            index (int): Index of the image to retrieve
            
        Returns:
            numpy.ndarray: 2D array with shape (1, height*width) ready for prediction
        """
        if dataset == 'train':
            if index >= len(self.trainSet):
                raise ValueError(f"Index {index} is out of range for training set (max: {len(self.trainSet)-1})")
            return self.trainSet[index].reshape(1, -1)
        elif dataset == 'test':
            if index >= len(self.testSet):
                raise ValueError(f"Index {index} is out of range for test set (max: {len(self.testSet)-1})")
            return self.testSet[index].reshape(1, -1)
        else:
            raise ValueError("Dataset must be either 'train' or 'test'")