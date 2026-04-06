"""
Utility functions for the Crop Recommendation System
Includes data processing, validation, and analysis helpers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataValidator:
    """Validate input data for crop recommendation"""
    
    @staticmethod
    def validate_inputs(N: float, P: float, K: float, 
                       temperature: float, humidity: float, 
                       ph: float, rainfall: float) -> Tuple[bool, str]:
        """
        Validate all input parameters
        
        Args:
            N: Nitrogen content
            P: Phosphorus content
            K: Potassium content
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            ph: Soil pH
            rainfall: Rainfall in mm
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        # Check NPK values
        if not (0 <= N <= 150):
            return False, "Nitrogen must be between 0-150"
        if not (0 <= P <= 150):
            return False, "Phosphorus must be between 0-150"
        if not (0 <= K <= 150):
            return False, "Potassium must be between 0-150"
        
        # Check environmental factors
        if not (0 <= temperature <= 50):
            return False, "Temperature must be between 0-50°C"
        if not (0 <= humidity <= 100):
            return False, "Humidity must be between 0-100%"
        if not (0 <= ph <= 14):
            return False, "pH must be between 0-14"
        if not (0 <= rainfall <= 300):
            return False, "Rainfall must be between 0-300mm"
        
        return True, "All inputs valid"

class CropAdvisor:
    """Provide detailed farming advice for recommended crops"""
    
    CROP_DATABASE = {
        "Rice": {
            "season": "Monsoon/Autumn",
            "ideal_temp": "20-25°C",
            "water_need": "High (150-250 mm)",
            "soil_type": "Clay/Loamy",
            "benefits": "Rich in carbohydrates, staple food",
            "tips": "Requires puddled fields and proper water management",
            "varieties": ["Basmati", "Jasmine", "Indica"],
            "fertilizer_schedule": {
                "Basal": "10 tons FYM + 150 kg NPK",
                "Top dressing 1": "60 kg N at 20-25 days",
                "Top dressing 2": "60 kg N at 50-60 days"
            }
        },
        "Wheat": {
            "season": "Winter/Spring",
            "ideal_temp": "15-25°C",
            "water_need": "Moderate (40-50 cm)",
            "soil_type": "Loamy/Well-drained",
            "benefits": "Protein-rich, long shelf-life",
            "tips": "Sow in winter, harvest in spring. Excellent for rotation",
            "varieties": ["HD 2967", "PBW 502", "HD 3086"],
            "fertilizer_schedule": {
                "Basal": "10 tons FYM + 100 kg NPK",
                "Top dressing": "80 kg N at 45 days"
            }
        },
        "Maize": {
            "season": "Summer/Monsoon",
            "ideal_temp": "21-27°C",
            "water_need": "Moderate-High (60 cm)",
            "soil_type": "Well-drained loam",
            "benefits": "Versatile, animal feed, industrial use",
            "tips": "Requires spacing and timely irrigation during grain filling",
            "varieties": ["DHM 117", "Bioseed 7777", "NK 6240"],
            "fertilizer_schedule": {
                "Basal": "10 tons FYM + 80 kg N + 40 kg P + 40 kg K",
                "Top dressing": "80 kg N at 35 days"
            }
        },
        "Cotton": {
            "season": "Spring/Summer",
            "ideal_temp": "20-30°C",
            "water_need": "High (6-8 irrigations)",
            "soil_type": "Black soil/Deep loam",
            "benefits": "Cash crop, textile industry",
            "tips": "Requires deep soil and excellent drainage",
            "varieties": ["GFH-18", "Ankur 3028", "RCH 2"],
            "fertilizer_schedule": {
                "Basal": "15 tons FYM + 120 kg NPK",
                "Top dressing 1": "60 kg N at 60 days",
                "Top dressing 2": "60 kg N at 120 days"
            }
        },
        "Sugarcane": {
            "season": "Year-round",
            "ideal_temp": "21-27°C",
            "water_need": "Very High (150-250 cm)",
            "soil_type": "Loamy/Alluvial",
            "benefits": "Sugar production, high returns",
            "tips": "Requires 12-18 months growth period",
            "varieties": ["CoH 119", "CoS 8436", "BO 91"],
            "fertilizer_schedule": {
                "Basal": "20 tons FYM + 150 kg N + 75 kg P",
                "Top dressing 1": "150 kg N at 90 days",
                "Top dressing 2": "150 kg N at 180 days"
            }
        },
        "Millet": {
            "season": "Summer/Monsoon",
            "ideal_temp": "25-35°C",
            "water_need": "Low (40-50 cm)",
            "soil_type": "Light sandy soil",
            "benefits": "Drought-resistant, nutritious",
            "tips": "Ideal for arid/semi-arid regions",
            "varieties": ["Pusa Composite 443", "HB 3", "JB 535"],
            "fertilizer_schedule": {
                "Basal": "5 tons FYM + 40 kg NPK",
                "Top dressing": "40 kg N at 25 days"
            }
        },
        "Pulses": {
            "season": "Winter/Summer",
            "ideal_temp": "20-30°C",
            "water_need": "Low-Moderate (40-60 cm)",
            "soil_type": "Well-drained",
            "benefits": "High protein, nitrogen-fixing",
            "tips": "Excellent for crop rotation",
            "varieties": ["Arjun", "Akanksha", "PUSA 256"],
            "fertilizer_schedule": {
                "Basal": "8 tons FYM + 20 kg N + 50 kg P"
            }
        },
        "Groundnut": {
            "season": "Summer/Monsoon",
            "ideal_temp": "24-28°C",
            "water_need": "Moderate (60-90 cm)",
            "soil_type": "Light sandy loam",
            "benefits": "Oil content, protein-rich",
            "tips": "Requires well-drained sandy soil",
            "varieties": ["TG 26", "JL 24", "M 13"],
            "fertilizer_schedule": {
                "Basal": "8 tons FYM + 25 kg N + 75 kg P"
            }
        },
        "Soybean": {
            "season": "Monsoon",
            "ideal_temp": "20-30°C",
            "water_need": "Moderate (60-70 cm)",
            "soil_type": "Well-drained loam",
            "benefits": "High protein, oil content",
            "tips": "Modern crop with market demand",
            "varieties": ["JS 97-52", "MACS 1407", "Super 7701"],
            "fertilizer_schedule": {
                "Basal": "8 tons FYM + 0 kg N + 60 kg P"
            }
        }
    }
    
    @classmethod
    def get_advice(cls, crop: str) -> Dict:
        """Get detailed advice for a crop"""
        return cls.CROP_DATABASE.get(crop, {})
    
    @classmethod
    def get_all_crops(cls) -> List[str]:
        """Get list of all available crops"""
        return list(cls.CROP_DATABASE.keys())

class SoilAnalyzer:
    """Analyze soil conditions and provide recommendations"""
    
    @staticmethod
    def analyze_npk(N: float, P: float, K: float) -> Dict:
        """Analyze NPK ratio and provide recommendations"""
        
        analysis = {
            "N": {"value": N, "status": "", "recommendation": ""},
            "P": {"value": P, "status": "", "recommendation": ""},
            "K": {"value": K, "status": "", "recommendation": ""}
        }
        
        # Nitrogen analysis
        if N < 40:
            analysis["N"]["status"] = "Low"
            analysis["N"]["recommendation"] = "Add nitrogen fertilizer (urea, ammonium nitrate)"
        elif 40 <= N <= 80:
            analysis["N"]["status"] = "Moderate"
            analysis["N"]["recommendation"] = "Moderate nitrogen level, suitable for many crops"
        else:
            analysis["N"]["status"] = "High"
            analysis["N"]["recommendation"] = "High nitrogen, good for leafy crops"
        
        # Phosphorus analysis
        if P < 30:
            analysis["P"]["status"] = "Low"
            analysis["P"]["recommendation"] = "Add phosphate fertilizer (DAP, SSP)"
        elif 30 <= P <= 60:
            analysis["P"]["status"] = "Moderate"
            analysis["P"]["recommendation"] = "Adequate phosphorus for most crops"
        else:
            analysis["P"]["status"] = "High"
            analysis["P"]["recommendation"] = "Sufficient phosphorus available"
        
        # Potassium analysis
        if K < 30:
            analysis["K"]["status"] = "Low"
            analysis["K"]["recommendation"] = "Add potassium fertilizer (MOP, SOP)"
        elif 30 <= K <= 60:
            analysis["K"]["status"] = "Moderate"
            analysis["K"]["recommendation"] = "Good potassium level for balanced growth"
        else:
            analysis["K"]["status"] = "High"
            analysis["K"]["recommendation"] = "High potassium level present"
        
        return analysis
    
    @staticmethod
    def analyze_ph(ph: float) -> Dict:
        """Analyze soil pH and provide recommendations"""
        
        if ph < 5.5:
            return {
                "status": "Very Acidic",
                "suitable_crops": ["Rice", "Sugarcane"],
                "recommendation": "Apply lime to increase pH"
            }
        elif 5.5 <= ph < 6.5:
            return {
                "status": "Acidic",
                "suitable_crops": ["Rice", "Sugarcane", "Maize"],
                "recommendation": "Slightly acidic; suitable for most crops"
            }
        elif 6.5 <= ph <= 7.5:
            return {
                "status": "Neutral",
                "suitable_crops": ["Wheat", "Pulses", "Groundnut", "Soybean"],
                "recommendation": "Ideal pH range for most crops"
            }
        elif 7.5 < ph < 8.5:
            return {
                "status": "Alkaline",
                "suitable_crops": ["Wheat", "Cotton", "Millet"],
                "recommendation": "Slightly alkaline; apply sulfur if needed"
            }
        else:
            return {
                "status": "Very Alkaline",
                "suitable_crops": ["Millet", "Groundnut"],
                "recommendation": "Apply acidifying amendments (sulfur, FeSO4)"
            }

class WeatherAnalyzer:
    """Analyze weather conditions for crop suitability"""
    
    @staticmethod
    def analyze_conditions(temperature: float, humidity: float, rainfall: float) -> Dict:
        """Analyze weather conditions"""
        
        analysis = {
            "temperature": {"value": temperature, "interpretation": ""},
            "humidity": {"value": humidity, "interpretation": ""},
            "rainfall": {"value": rainfall, "interpretation": ""}
        }
        
        # Temperature interpretation
        if temperature < 15:
            analysis["temperature"]["interpretation"] = "Too cold for tropical crops"
        elif 15 <= temperature < 20:
            analysis["temperature"]["interpretation"] = "Cool - suitable for winter crops"
        elif 20 <= temperature <= 30:
            analysis["temperature"]["interpretation"] = "Optimal - suitable for most crops"
        else:
            analysis["temperature"]["interpretation"] = "Very hot - requires heat-tolerant crops"
        
        # Humidity interpretation
        if humidity < 40:
            analysis["humidity"]["interpretation"] = "Low humidity - water-efficient crops needed"
        elif 40 <= humidity < 60:
            analysis["humidity"]["interpretation"] = "Moderate humidity - good for most crops"
        elif 60 <= humidity <= 80:
            analysis["humidity"]["interpretation"] = "High humidity - suitable for wet crops"
        else:
            analysis["humidity"]["interpretation"] = "Very high - disease-prone conditions"
        
        # Rainfall interpretation
        if rainfall < 40:
            analysis["rainfall"]["interpretation"] = "Low rainfall - drought-tolerant crops needed"
        elif 40 <= rainfall < 100:
            analysis["rainfall"]["interpretation"] = "Moderate rainfall - suitable for semi-arid crops"
        elif 100 <= rainfall < 200:
            analysis["rainfall"]["interpretation"] = "Good rainfall - suitable for most crops"
        else:
            analysis["rainfall"]["interpretation"] = "High rainfall - suitable for water-loving crops"
        
        return analysis

def create_sample_data() -> pd.DataFrame:
    """Create sample dataset for model training"""
    
    data = {
        "N": [90, 85, 60, 74, 78, 69, 50, 40, 60, 80, 92, 75, 55, 85, 70],
        "P": [42, 58, 55, 35, 40, 37, 45, 50, 48, 60, 55, 40, 50, 45, 52],
        "K": [43, 41, 44, 40, 42, 38, 35, 30, 33, 45, 44, 42, 32, 46, 39],
        "temperature": [20, 25, 30, 22, 26, 28, 32, 34, 27, 23, 21, 29, 33, 24, 25],
        "humidity": [82, 80, 75, 85, 70, 65, 60, 55, 72, 78, 83, 68, 62, 77, 71],
        "ph": [6.5, 7.0, 6.8, 6.2, 6.9, 7.1, 6.3, 6.0, 6.7, 6.4, 6.6, 7.0, 6.1, 6.8, 6.5],
        "rainfall": [200, 180, 150, 210, 170, 160, 140, 120, 155, 190, 195, 165, 135, 205, 175],
        "crop": ["Rice", "Wheat", "Maize", "Rice", "Wheat", "Maize", "Cotton", 
                "Millet", "Sugarcane", "Rice", "Wheat", "Maize", "Millet", "Rice", "Cotton"]
    }
    
    return pd.DataFrame(data)
