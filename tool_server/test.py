import base64
import io

import requests
from imgcat import imgcat
from PIL import Image


def test_detection_api(image_path, classes, confidence=0.1):
    """Test the object detection API with a local image."""
    # API endpoint 
    url = f"http://localhost:8000/detect?confidence={confidence}"
    
    # Prepare the image file
    files = {"image": open(image_path, "rb")}
    
    # Send each class as a separate form field with the same name
    data = {}
    for c in classes:
        data.setdefault("classes", []).append(c)
    
    # Send request
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract detection results
        detections = result["results"]["detections"]
        
        print(f"Detected {len(detections)} objects:")
        for detection in detections:
            print(f"  {detection['label']}: {detection['confidence']:.2f} at {detection['bbox_2d']}")
        
        # Display the annotated image
        img_data = base64.b64decode(result["annotated_image"])
        img = Image.open(io.BytesIO(img_data))
        imgcat(img)
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    # Example usage
    image_path = "cars.jpeg"  
    classes = ["red car", "telephone pole"] 
    test_detection_api(image_path, classes)
