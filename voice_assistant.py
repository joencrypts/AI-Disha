"""
Voice Assistant module for AI-Powered Proactive Patient Risk Advisor
Integrates with ElevenLabs for natural language interaction
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class VoiceAssistant:
    """Voice assistant for patient interaction using ElevenLabs"""
    
    def __init__(self, config):
        self.config = config
        self.elevenlabs_config = config['ELEVENLABS_CONFIG']
        self.api_key = self.elevenlabs_config['api_key']
        self.voice_id = self.elevenlabs_config['voice_id']
        self.model_id = self.elevenlabs_config['model_id']
        self.base_url = "https://api.elevenlabs.io/v1"
        
    def generate_speech(self, text: str, voice_id: str = None) -> bytes:
        """Generate speech from text using ElevenLabs API"""
        if not self.api_key:
            print("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY environment variable.")
            return None
        
        voice_id = voice_id or self.voice_id
        
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error generating speech: {e}")
            return None
    
    def save_audio(self, audio_data: bytes, filename: str) -> str:
        """Save audio data to file"""
        if audio_data is None:
            return None
        
        # Create audio directory if it doesn't exist
        audio_dir = self.config['STATIC_DIR'] / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        file_path = audio_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        
        return str(file_path)
    
    def create_patient_interaction_script(self, condition_type: str) -> str:
        """Create interaction script for patient consultation"""
        
        scripts = {
            'kidney': """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your kidney health risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. Have you been experiencing any changes in your urination patterns recently?
            2. Do you have any swelling in your feet, ankles, or hands?
            3. Have you noticed any changes in your energy levels or fatigue?
            4. Are you currently taking any medications for blood pressure or diabetes?
            5. Do you have a family history of kidney disease?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """,
            'general': """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your general health risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. How would you describe your current overall health?
            2. Do you have any chronic conditions or ongoing health concerns?
            3. What is your current lifestyle regarding diet and exercise?
            4. Do you have any family history of health conditions?
            5. Are you taking any medications or supplements?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """,
            
            'diabetes': """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your diabetes risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. Have you been experiencing increased thirst or frequent urination?
            2. Have you noticed any unexplained weight changes recently?
            3. Do you have any family history of diabetes?
            4. How would you describe your current diet and exercise routine?
            5. Have you had any recent blood sugar tests?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """,
            
            'heart': """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your cardiovascular health risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. Have you been experiencing any chest pain or discomfort?
            2. Do you have any shortness of breath during normal activities?
            3. Have you noticed any irregular heartbeats or palpitations?
            4. Do you have a family history of heart disease?
            5. What is your current lifestyle regarding smoking, diet, and exercise?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """,
            
            'liver': """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your liver health risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. Have you been experiencing any abdominal pain or discomfort?
            2. Have you noticed any yellowing of your skin or eyes?
            3. Do you have any changes in your appetite or weight?
            4. Have you been taking any medications or supplements recently?
            5. Do you have a history of alcohol consumption or liver disease?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """
        }
        
        return scripts.get(condition_type, scripts['general'])
    
    def generate_patient_response(self, patient_symptoms: str, 
                                risk_assessment: Dict[str, Any]) -> str:
        """Generate personalized response based on patient symptoms and risk assessment"""
        
        risk_level = risk_assessment.get('final_risk_level', 'UNKNOWN')
        risk_probability = risk_assessment.get('final_risk_probability', 0.5)
        recommended_actions = risk_assessment.get('recommended_actions', [])
        
        response = f"""
        Thank you for sharing your symptoms and concerns. Based on my analysis, here's what I found:
        
        RISK ASSESSMENT:
        Your current risk level is {risk_level}. The probability of developing this condition is {risk_probability:.1%}.
        
        """
        
        if risk_level == 'HIGH':
            response += """
        ⚠️ HIGH RISK ALERT ⚠️
        Your symptoms and risk factors indicate a high risk. I strongly recommend:
        - Seeking immediate medical attention
        - Consulting with a specialist
        - Following up with your primary care physician
        """
        elif risk_level == 'MODERATE':
            response += """
        ⚠️ MODERATE RISK
        Your symptoms suggest moderate risk. I recommend:
        - Scheduling an appointment with your doctor
        - Monitoring your symptoms closely
        - Making lifestyle modifications
        """
        else:
            response += """
        ✅ LOW RISK
        Your current risk appears to be low. Continue with:
        - Regular health monitoring
        - Preventive care measures
        - Healthy lifestyle maintenance
        """
        
        if recommended_actions:
            response += "\n\nRECOMMENDED ACTIONS:\n"
            for i, action in enumerate(recommended_actions[:5], 1):
                response += f"{i}. {action}\n"
        
        response += """
        
        IMPORTANT DISCLAIMER:
        This assessment is for informational purposes only and should not replace professional medical advice. 
        Please consult with a qualified healthcare provider for proper diagnosis and treatment.
        
        Would you like me to explain any of these recommendations in more detail?
        """
        
        return response
    
    def create_voice_interaction(self, condition_type: str, 
                               patient_symptoms: str = None,
                               risk_assessment: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create complete voice interaction"""
        
        # Generate initial script
        initial_script = self.create_patient_interaction_script(condition_type)
        
        # Generate initial audio
        initial_audio = self.generate_speech(initial_script)
        initial_audio_path = None
        if initial_audio:
            initial_audio_path = self.save_audio(initial_audio, f"{condition_type}_initial.mp3")
        
        result = {
            'condition_type': condition_type,
            'initial_script': initial_script,
            'initial_audio_path': initial_audio_path,
            'patient_symptoms': patient_symptoms,
            'risk_assessment': risk_assessment
        }
        
        # Generate response if patient symptoms and risk assessment provided
        if patient_symptoms and risk_assessment:
            response_script = self.generate_patient_response(patient_symptoms, risk_assessment)
            response_audio = self.generate_speech(response_script)
            response_audio_path = None
            if response_audio:
                response_audio_path = self.save_audio(response_audio, f"{condition_type}_response.mp3")
            
            result.update({
                'response_script': response_script,
                'response_audio_path': response_audio_path
            })
        
        return result
    
    def create_emergency_alert(self, patient_data: Dict[str, Any], 
                             risk_assessment: Dict[str, Any]) -> str:
        """Create emergency alert for high-risk patients"""
        
        risk_level = risk_assessment.get('final_risk_level', 'UNKNOWN')
        
        if risk_level != 'HIGH':
            return None
        
        alert_script = f"""
        🚨 EMERGENCY ALERT 🚨
        
        HIGH RISK PATIENT DETECTED
        
        Patient ID: {patient_data.get('patient_id', 'Unknown')}
        Risk Level: {risk_level}
        Risk Probability: {risk_assessment.get('final_risk_probability', 0):.1%}
        
        IMMEDIATE ACTIONS REQUIRED:
        1. Contact patient immediately
        2. Schedule urgent medical consultation
        3. Consider emergency department referral
        4. Notify primary care physician
        
        This is an automated alert from the AI-Powered Proactive Patient Risk Advisor system.
        """
        
        return alert_script
    
    def generate_voice_alert(self, alert_script: str) -> str:
        """Generate voice alert for emergency situations"""
        if not alert_script:
            return None
        
        # Use a more urgent voice for alerts
        alert_audio = self.generate_speech(alert_script, voice_id="pNInz6obpgDQGcFmaJgB")
        
        if alert_audio:
            alert_path = self.save_audio(alert_audio, "emergency_alert.mp3")
            return alert_path
        
        return None
    
    def create_interaction_summary(self, interaction_data: Dict[str, Any]) -> str:
        """Create summary of voice interaction"""
        
        summary = f"""
        VOICE INTERACTION SUMMARY
        
        Condition Type: {interaction_data.get('condition_type', 'Unknown')}
        Patient Symptoms: {interaction_data.get('patient_symptoms', 'Not provided')}
        
        Risk Assessment:
        - Risk Level: {interaction_data.get('risk_assessment', {}).get('final_risk_level', 'Unknown')}
        - Risk Probability: {interaction_data.get('risk_assessment', {}).get('final_risk_probability', 0):.1%}
        
        Audio Files Generated:
        - Initial Script: {interaction_data.get('initial_audio_path', 'Not generated')}
        - Response: {interaction_data.get('response_audio_path', 'Not generated')}
        
        Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return summary
