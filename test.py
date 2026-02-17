import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os
import json
import re
from gtts import gTTS
import tempfile

# Load environment variables from a .env file
load_dotenv()

# Configure the GenerativeAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Language to TTS code mapping
LANGUAGE_TTS_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
}

def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

def get_base_analysis(image_data):
    """Get base analysis in English to determine accuracy"""
    base_prompt = """
    Analyze this plant image in detail. Focus on:
    1. Plant identification (species and family)
    2. Disease identification
    3. Image quality assessment
    4. Symptom clarity
    5. Confidence in diagnosis
    
    Provide the response in this exact JSON format:
    {
        "plant_name": "common and scientific name",
        "plant_family": "family name",
        "disease_name": "disease with scientific name",
        "accuracy_score": "numerical score 0-100 based on image quality and confidence",
        "confidence_notes": "explanation of accuracy",
        "symptoms": ["list", "of", "symptoms"],
        "treatments": ["list", "of", "treatments"],
        "prevention": ["list", "of", "measures"]
    }
    """
    
    try:
        response = model.generate_content([base_prompt, image_data])
        # Extract JSON from response
        json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
        analysis_data = json.loads(json_str)
        return analysis_data
    except Exception as e:
        print(f"Error in base analysis: {str(e)}")
        return None

def translate_analysis(analysis_data, language):
    """Translate the analysis to the target language while maintaining the accuracy"""
    if not analysis_data:
        return "Error in analysis", 70.0

    translation_prompt = f"""
    Translate the following plant disease analysis to {language}, maintaining the exact same format:

    Plant Analysis Results:

    Plant Identification:
    - Plant Name: {analysis_data['plant_name']}
    - Plant Family: {analysis_data['plant_family']}

    Disease Analysis:
    1. Disease Identification: {analysis_data['disease_name']}

    Prediction Accuracy: {analysis_data['accuracy_score']}%
    Confidence Assessment: {analysis_data['confidence_notes']}

    2. Symptoms Observed:
    {', '.join(analysis_data['symptoms'])}

    3. Recommended Treatments:
    {', '.join(analysis_data['treatments'])}

    4. Prevention Measures:
    {', '.join(analysis_data['prevention'])}
    """

    try:
        response = model.generate_content(translation_prompt)
        return response.text, float(analysis_data['accuracy_score'])
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return str(e), 70.0

def clean_kannada_text(text):
    """Clean and format Kannada text for better readability"""
    # Remove unwanted special characters
    text = re.sub(r'[\*\.,#]', '', text)
    # Replace percentages with words
    text = re.sub(r'(\d+)%', r'\1 ಶೇಕಡಾ', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    # Ensure proper line breaks and spacing
    text = text.replace(":", ":\n")
    text = text.replace("- ", "\n- ")
    text = text.replace("1. ", "\n1. ")
    text = text.replace("2. ", "\n2. ")
    text = text.replace("3. ", "\n3. ")
    text = text.replace("4. ", "\n4. ")
    return text.strip()

def clean_text_for_speech(text, language):
    """Clean and format text for TTS, removing English words for non-English languages"""
    # Remove unwanted characters
    text = re.sub(r'[\*\.,#]', '', text)
    # Replace percentages with words
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # For non-English languages, remove English words
    if language != "English":
        # Remove text in parentheses (e.g., scientific names)
        text = re.sub(r'\([^)]*\)', '', text)
        # Remove standalone English words
        text = re.sub(r'\b[A-Za-z]+\b', '', text)
    
    return text.strip()

def text_to_speech(text, language):
    """Convert text to speech using gTTS and save to a temporary file"""
    try:
        cleaned_text = clean_text_for_speech(text, language)
        print(f"Cleaned text for TTS: {cleaned_text}")  # Debugging
        tts = gTTS(
            text=cleaned_text,
            lang=LANGUAGE_TTS_CODES[language],
            slow=False
        )
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            print(f"TTS audio saved to: {temp_file.name}")  # Debugging
            return temp_file.name
    except Exception as e:
        print(f"Error in TTS: {str(e)}")
        return None

def generate_gemini_response_with_accuracy(image_path, language):
    try:
        image_data = read_image_data(image_path)
        
        # First, get base analysis in English
        analysis_data = get_base_analysis(image_data)
        
        # Then translate to target language while maintaining the accuracy
        translated_response, accuracy = translate_analysis(analysis_data, language)
        
        return translated_response, accuracy
    except Exception as e:
        return f"Error during analysis: {str(e)}", 70.0

def process_uploaded_files_with_accuracy(files, language):
    file_path = files[0].name if files else None
    if file_path:
        # First, generate and display the text analysis
        response, accuracy = generate_gemini_response_with_accuracy(file_path, language)
        
        # Clean the Kannada text for better readability
        if language == "Kannada":
            response = clean_kannada_text(response)
        
        # Return the image, text analysis, and a placeholder for the audio
        return file_path, response
    
    return None, "No file uploaded"

def generate_tts_audio(response, language):
    """Generate TTS audio when the user clicks the button"""
    if response and language:
        print(f"Generating TTS for language: {language}")  # Debugging
        audio_path = text_to_speech(response, language)
        if audio_path:
            print(f"TTS audio generated at: {audio_path}")  # Debugging
            return audio_path
        else:
            print("TTS audio generation failed.")  # Debugging
            return None
    return None

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Plant Disease Detection with Plant Identification")
    
    # Language selection
    language_select = gr.Dropdown(
        choices=["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam"],
        value="English",
        label="Select Language"
    )
    
    file_output = gr.Textbox(label="Analysis", lines=12)
    image_output = gr.Image(label="Uploaded Image")
    audio_output = gr.Audio(label="Speech Output", type="filepath", visible=True)  # Make audio component visible
    tts_button = gr.Button("Generate Audio", variant="primary")
    combined_output = [image_output, file_output, audio_output, tts_button]

    # Upload button
    upload_button = gr.UploadButton(
        "Upload Plant Image",
        file_types=["image"],
        file_count="multiple",
    )
    
    # Handle upload with language
    upload_button.upload(
        process_uploaded_files_with_accuracy, 
        inputs=[upload_button, language_select], 
        outputs=[image_output, file_output]
    )
    
    # Generate TTS audio when the button is clicked
    tts_button.click(
        generate_tts_audio,
        inputs=[file_output, language_select],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch(debug=True)