import google.generativeai as genai
from PIL import Image
import io

class AgriSmartAI:
    """
    Central AI Brain for the AgriSmart Dashboard.
    Handles all interactions with the Google Gemini API.
    """
    def __init__(self):
        # Your specific Gemini API Key
        self.api_key = "AIzaSyDl6pL8DFAVh8fYClswbTTOb2J7XpDEmW8"
        genai.configure(api_key=self.api_key)
        
        # We use gemini-2.5-flash as it supports both text and vision natively
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=(
                "You are AgriSmart AI, an expert agricultural assistant dedicated to helping farmers in India. "
                "You provide practical, accurate, and easy-to-understand advice regarding crops, "
                "diseases, soil management, and government schemes. Keep answers concise and clear, as they will often be spoken out loud."
            )
        )

    def process_voice(self, user_text, language="English"):
        """Handles the multilingual Voice Assistant logic"""
        prompt = f"The farmer asked: '{user_text}'. Answer concisely and helpfully. You MUST reply completely in the {language} language."
        response = self.model.generate_content(prompt)
        return response.text

    def analyze_disease(self, image_bytes):
        """Handles the Computer Vision logic for Crop Diseases"""
        # Convert raw bytes into a format the AI understands
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = (
            "Analyze this plant leaf/crop image.\n"
            "1. Identify the exact Crop Name.\n"
            "2. Identify the specific disease, pest, or problem.\n"
            "3. Provide a step-by-step actionable solution or treatment plan to cure it."
        )
        response = self.model.generate_content([prompt, img])
        return response.text

    def analyze_soil(self, lat, lng, image_bytes=None):
        """Handles Soil matching using Vision and Location data"""
        base_prompt = f"I am a farmer at GPS coordinates Lat: {lat}, Lng: {lng}. "
        
        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes))
            prompt = base_prompt + "Based on this location's climate and the attached visual soil image, determine the soil type and recommend the top 3 best crops to grow right now. Explain why."
            response = self.model.generate_content([prompt, img])
        else:
            prompt = base_prompt + "Based on this geographical location's typical soil and climate, recommend the top 3 best crops to grow right now and explain why."
            response = self.model.generate_content(prompt)
            
        return response.text

    def get_schemes(self, query):
        """Handles fetching Government Schemes"""
        prompt = f"List 3 specific, real, active government agricultural schemes or subsidies related to: '{query}'. Provide the Name, Category, and a brief Description for each."
        response = self.model.generate_content(prompt)
        return response.text

    def run_digital_twin(self, csv_data_string):
        """Handles Predictive Analysis on historical farm data"""
        prompt = f"""Here is the historical data of my farm: \n{csv_data_string}\n
        Act as a Farm Digital Twin AI. Analyze this data deeply. Predict the yield for the next year assuming standard rainfall. 
        Provide a detailed explanation of your prediction and suggest 2 preventative measures to ensure maximum yield."""
        
        response = self.model.generate_content(prompt)
        return response.text