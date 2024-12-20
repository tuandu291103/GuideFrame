import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
import os
import pytesseract
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
import requests
import os
from openai import OpenAI

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

# Print the extracted text
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

# Function to summarize text using OpenAI API
def summarize_text(text):
    #api_key = "sk"
    api_key ="insert your api key"
    api_endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Simplified format for image-ready summarization
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a helpful assistant that summarizes medical instructions in a concise, simplified form. "
                "Focus on clear English words, using very short phrases or single words where possible. Only include essential information "
                "like dosage, frequency, warnings, and any specific instructions for visualization. Make it suitable as a guide for visual interpretation."
            )
        },
        {
            "role": "user", 
            "content": (
                "Create a very simplified version of the following instructions, focusing only on key points and avoiding unnecessary details. "
                "Use clear, short phrases in English that would make sense as labels or captions in an instructional image:\n\n" + text
            )
        }
    ]
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 100,  # Reduced token count for brief output
        "temperature": 0.2
    }

    response = requests.post(api_endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        summary = response.json()["choices"][0]["message"]["content"].strip()
        return summary
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

'''
# Function to send text to the image generation API
def generate_image_from_text(text):
    # Replace with your actual API endpoint
    api_endpoint = "https://api.openai.com/v1/images/generations"  # Replace with actual endpoint
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    prompt = f"Create a clear visual guide with labeled instructions: {text}"
    payload = {
        "prompt": prompt,
        "size": "1024x1024"
    }

    response = requests.post(api_endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        image_url = response.json().get("data")[0]["url"]
        return image_url
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
'''
# Keyword extraction for image selection
def extract_keywords(text):
    # Simple keyword extraction based on common medical terms, dosage, etc.
    keywords = re.findall(r'\b(dosage|frequency|apply|avoid|drink|once|daily|mg|tablet)\b', text.lower())
    return list(set(keywords))

# Select image based on keywords
def select_image_based_on_keywords(keywords):
    # Sample mapping of keywords to image filenames
    image_map = {
        "dosage": "static/images/output1.png",
        "frequency": "static/images/output1.png",
        "apply": "static/images/output2.png",
        # Add more mappings as needed
    }
    for keyword in keywords:
        if keyword in image_map:
            return image_map[keyword]
    return "static/images/output2.png"  # Fallback image


''' 
# Use sample text or get text from a form input in your HTML
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form.get('text') or "Sample text for summarization."
    summarized_text = summarize_text(text)
    if summarized_text:
        keywords = extract_keywords(summarized_text)  # Extract keywords from summarized text
        image_url = select_image_based_on_keywords(keywords)  # Select appropriate image based on keywords
        return render_template('index.html', image_url=image_url, summarized_text=summarized_text)
    else:
        return render_template('index.html', error="Failed to summarize text.")
'''
# Use sample text or get text from a form input in your HTML
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/generate', methods=['POST'])
def generate():
    text_input = request.form.get('text')
    file = request.files.get('file')

    # Ensure text or file is provided
    if not text_input and not file:
        return render_template('index.html', error="Please provide either text or a PDF file.")

    # Handle PDF input
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            pdf_reader = PdfReader(filepath)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            input_text = pdf_text
        except Exception as e:
            return render_template('index.html', error="Failed to extract text from PDF.")
    else:
        # Use text input if provided
        input_text = text_input

    # Summarize the text
    summarized_text = summarize_text(input_text)
    if summarized_text:
        keywords = extract_keywords(summarized_text)  # Extract keywords from summarized text
        image_url = select_image_based_on_keywords(keywords)  # Select appropriate image based on keywords
        return render_template('index.html', image_url=image_url, summarized_text=summarized_text)
    else:
        return render_template('index.html', error="Failed to summarize text.")



if __name__ == '__main__':
    app.run(debug=True)

# Send the cleaned text to the image generation API
#image_url = generate_image_from_text(cleaned_text)
#if image_url:
#   print(f"Generated Image URL: {image_url}")




    
