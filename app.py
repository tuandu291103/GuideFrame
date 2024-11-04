import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
import os
import pytesseract



# Add this line before you call any pytesseract functions
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Load the image using OpenCV
image = cv2.imread('medical_instruction.png')

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found.")
    exit()
    
 # Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply noise reduction (optional)
gray = cv2.medianBlur(gray, 3)

# Apply thresholding to get a binary image
gray = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 2
)

# Convert the processed image to a PIL Image
pil_image = Image.fromarray(gray)

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(pil_image)

print("Extracted Text:")
print(text)

def clean_text(text):
    # Remove non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode('utf-8')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

cleaned_text = clean_text(text)

print("\nCleaned Text:")
print(cleaned_text)

with open('output_description.txt', 'w') as file:
    file.write(cleaned_text)
    
# Function to send text to the image generation API
def generate_image_from_text(text):
    # Replace with your actual API endpoint
    api_endpoint = "https://api.imagegen-service.com/v1/generate"  # Replace with actual endpoint
    headers = {
        "Authorization": f"Bearer {os.getenv('IMAGEGEN_API_KEY')}",  # Make sure IMAGEGEN_API_KEY is set
        "Content-Type": "application/json"
    }
    
    # Format the text as a prompt
    prompt = f"Create a visual guide for: {text}"
    
    # Set up the payload
    payload = {
        "prompt": prompt,
        "size": "1024x1024"  # Define the image size if required by the API
    }
    
    # Make the request
    response = requests.post(api_endpoint, headers=headers, json=payload)
    
    # Check the response
    if response.status_code == 200:
        image_url = response.json().get("url")
        print(f"Image generated successfully: {image_url}")
        return image_url
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Send the cleaned text to the image generation API
image_url = generate_image_from_text(cleaned_text)
if image_url:
    print(f"Generated Image URL: {image_url}")



    
