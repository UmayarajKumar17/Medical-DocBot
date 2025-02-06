import os
import io
from PIL import Image
import google.generativeai as genai
import fitz  # PyMuPDF for text extraction
from pdf2image import convert_from_path  # Convert PDF pages to images
import streamlit as st

# Configure Gemini API (Replace with your API Key)
genai.configure(api_key="")

def extract_pdf_text(pdf_path):
    """Extract text from a PDF."""
    doc = fitz.open(pdf_path)  # Open PDF
    text_data = [page.get_text("text") for page in doc]  # Extract text
    return "\n".join(text_data)  # Combine all text

def extract_pdf_images(pdf_path):
    """Extract images from a PDF."""
    doc = fitz.open(pdf_path)  # Open PDF
    images = []

    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)

    return images  # List of images

def ask_gemini_llm(pdf_text, question):
    """Ask a question to Gemini Pro (Text-only LLM)."""
    model = genai.GenerativeModel("gemini-1.5-pro")  # Text LLM
    response = model.generate_content([pdf_text, question])
    return response.text  # Return LLM response

def analyze_images_with_vlm(images):
    """Analyze images using Gemini Vision (VLM)."""
    if not images:
        return "No images found in the PDF."

    model = genai.GenerativeModel("gemini-2.0-flash-001")  

    image_parts = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format)
        img_byte_arr = img_byte_arr.getvalue()

        image_parts.append({
            "mime_type": "image/" + img.format.lower(),  
            "data": img_byte_arr
        })
        
    response = model.generate_content(["Analyze these images and summarize their content:"] + image_parts)
    return response.text  

# Streamlit file uploader
st.title("PDF & Image Analyzer")

uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check if 'temp' directory exists, create it if not
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # If the file is a PDF
    if uploaded_file.type == "application/pdf":
        # Save the uploaded PDF file temporarily
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text and images from PDF
        pdf_text = extract_pdf_text(pdf_path)
        pdf_images = extract_pdf_images(pdf_path)

        image_analysis = analyze_images_with_vlm(pdf_images)

        # Display extracted text
        st.subheader("Extracted Text from PDF")
        st.text(pdf_text)

        # Display image analysis
        st.subheader("Image Analysis (VLM)")
        st.text(image_analysis)

        # Provide a link to download the PDF
        st.download_button(
            label="Download PDF",
            data=uploaded_file,
            file_name=uploaded_file.name,
            mime=uploaded_file.type
        )

    # If the file is an image
    elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Analyze the image using Gemini Vision (VLM)
        image_analysis = analyze_images_with_vlm([img])

        # Display image analysis
        st.subheader("Image Analysis (VLM)")
        st.text(image_analysis)

        # Provide a link to download the image
        st.download_button(
            label="Download Image",
            data=uploaded_file,
            file_name=uploaded_file.name,
            mime=uploaded_file.type
        )

    user_question = st.text_input("Ask a question about the PDF or Image:")
    if user_question:
        text_answer = ask_gemini_llm(pdf_text, user_question) if uploaded_file.type == "application/pdf" else "No PDF loaded for text-based questions"
        st.subheader("Text-Based Answer:")
        st.text(text_answer)
