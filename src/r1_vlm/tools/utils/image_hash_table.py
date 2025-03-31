import numpy as np
import PIL
from typing import TypeVar, Dict, Any, Optional

T = TypeVar('T')  # Type variable for the data associated with images

class ImageHashTable:
    """
    A generic hash table that maps images to arbitrary data.
    Uses a consistent random matrix for hashing images.
    """
    def __init__(self):
        self.hash_table: Dict[float, T] = {}
        self.hash_matrix: Optional[np.ndarray] = None
    
    def generate_hash_matrix(self, shape: tuple[int, int, int]) -> None:
        """
        Generates a hash matrix for the image. Saves to self.hash_matrix.
        Only generates if not already generated.
        
        Args:
            shape: The shape of the images that will be hashed (height, width, channels)
        """
        if self.hash_matrix is not None:
            raise ValueError("Error: Hash matrix already generated.")
        
        # Matrix of random floats between 0 and 1
        self.hash_matrix = np.random.random(shape)
    
    def hash_image(self, image: PIL.Image.Image) -> float:
        """
        Hashes the image using the hash matrix.
        
        Args:
            image: PIL Image to hash
            
        Returns:
            float: Hash value for the image
        """
        image_array = np.array(image)
        
        if self.hash_matrix is None:
            self.generate_hash_matrix(image_array.shape)
        
        # Hash the image using element-wise product sum
        hash_value = np.sum(image_array * self.hash_matrix)
        return hash_value
    
    def add_image(self, image: PIL.Image.Image, data: T) -> None:
        """
        Associates data with an image in the hash table.
        
        Args:
            image: PIL Image to store
            data: Data to associate with the image
        """
        hash_value = self.hash_image(image)
        self.hash_table[hash_value] = data
    
    def lookup_image(self, image: PIL.Image.Image) -> T:
        """
        Retrieves data associated with an image.
        
        Args:
            image: PIL Image to look up
            
        Returns:
            The data associated with the image
            
        Raises:
            ValueError: If the image is not found in the hash table
        """
        hash_value = self.hash_image(image)
        
        if hash_value not in self.hash_table:
            # Print and raise, as the value error gets swallowed during training.
            print(f"Error: Image not found in the hash table.")
            raise ValueError("Error: Image not found in the hash table.")
        
        return self.hash_table[hash_value]
