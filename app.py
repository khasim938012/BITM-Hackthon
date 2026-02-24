from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import pandas as pd
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # Allow frontend to communicate with backend

class AgriSmartAI:
    """Central AI Assistant class handling all Gemini interactions"""
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # We use system_instruction to give the AI a strong, persistent persona
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=(
                "You are AgriSmart AI, an expert agricultural assistant dedicated to helping farmers. "
                "You provide practical, accurate, and easy-to-understand advice regarding crops, "
                "diseases, soil management, and government schemes. Always be polite, encouraging, and clear."
            )
        )

    def process_voice_command(self, user_text, language):
        """Handles multilingual voice queries"""
        prompt = f"The farmer asked: '{user_text}'. Answer concisely and helpfully. You MUST reply completely in the {language} language."
        response = self.model.generate_content(prompt)
        return response.text

    def analyze_crop_disease(self, image_bytes):
        """Analyzes plant images for diseases and solutions"""
        img = Image.open(io.BytesIO(image_bytes))
        prompt = (
            "Analyze this plant image.\n"
            "1. Identify the crop name.\n"
            "2. Identify the specific disease or problem.\n"
            "3. Provide a step-by-step actionable solution to cure it."
        )
        response = self.model.generate_content([prompt, img])
        return response.text

    def match_soil_and_crops(self, lat, lng, image_bytes=None):
        """Recommends crops based on location and optional soil image"""
        base_prompt = f"I am a farmer at GPS coordinates Lat: {lat}, Lng: {lng}. "
        
        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes))
            prompt = base_prompt + "Based on this geographical location's typical climate and the attached soil image, determine the soil type and recommend the top 3 best crops to grow. Explain why."
            response = self.model.generate_content([prompt, img])
        else:
            prompt = base_prompt + "Based on this geographical location's typical soil and climate, recommend the top 3 best crops to grow and explain why."
            response = self.model.generate_content(prompt)
            
        return response.text

    def find_schemes(self, query):
        """Finds relevant government agricultural schemes"""
        prompt = (
            f"List 3 specific, real, active government agricultural schemes or subsidies related to: '{query}'. "
            "Provide the Name, Category, and Description for each."
        )
        response = self.model.generate_content(prompt)
        return response.text

    def simulate_digital_twin(self, csv_data_string):
        """Predicts future yields based on historical farm data"""
        prompt = f"""Here is the historical data of a farm: \n{csv_data_string}\n
        Act as a Farm Digital Twin AI. Analyze this data. Predict the yield for next year assuming standard rainfall. 
        Provide a deep explanation of your prediction and suggest preventative measures."""
        response = self.model.generate_content(prompt)
        return response.text


# Initialize the AI Brain with the API key
api_key = "AIzaSyDl6pL8DFAVh8fYClswbTTOb2J7XpDEmW8"
agri_ai = AgriSmartAI(api_key)


# --- FLASK ROUTES ---

@app.route('/api/voice', methods=['POST'])
def process_voice():
    data = request.json
    user_text = data.get('text', '')
    language = data.get('language', 'English')
    
    try:
        reply = agri_ai.process_voice_command(user_text, language)
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/disease', methods=['POST'])
def analyze_disease():
    data = request.json
    image_base64 = data.get('image') 
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
        
    try:
        image_bytes = base64.b64decode(image_base64)
        analysis = agri_ai.analyze_crop_disease(image_bytes)
        return jsonify({"message": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/soil', methods=['POST'])
def analyze_soil():
    data = request.json
    image_base64 = data.get('image')
    lat = data.get('lat', 'Unknown')
    lng = data.get('lng', 'Unknown')
    
    try:
        image_bytes = base64.b64decode(image_base64) if image_base64 else None
        analysis = agri_ai.match_soil_and_crops(lat, lng, image_bytes)
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/schemes', methods=['POST'])
def get_schemes():
    data = request.json
    query = data.get('query', 'Agricultural subsidies')
    
    try:
        schemes = agri_ai.find_schemes(query)
        return jsonify({"schemes": schemes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/twin', methods=['GET'])
def run_digital_twin():
    try:
        df = pd.read_csv('farm_data.csv')
        csv_string = df.to_csv(index=False)
        
        analysis = agri_ai.simulate_digital_twin(csv_string)
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)