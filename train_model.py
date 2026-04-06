"""
Model training and evaluation script for Crop Recommendation System
Run this script to train, evaluate, and save the model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
from pathlib import Path

class ModelTrainer:
    """Train and evaluate crop recommendation models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_data = None
        
    def create_training_data(self):
        """Create comprehensive training dataset"""
        
        # Expanded dataset with 45 samples covering 9 crops
        data = {
            "N": [90, 85, 60, 74, 78, 69, 50, 40, 60, 80, 92, 75, 55, 85, 70, 
                  88, 62, 95, 68, 72, 83, 65, 58, 79, 87, 61, 52, 84, 76, 91,
                  67, 73, 56, 81, 77, 89, 64, 59, 86, 71, 93, 70, 54, 82, 80],
            "P": [42, 58, 55, 35, 40, 37, 45, 50, 48, 60, 55, 40, 50, 45, 52,
                  58, 48, 44, 46, 53, 56, 42, 49, 39, 52, 51, 46, 43, 47, 60,
                  54, 38, 47, 41, 55, 59, 49, 44, 52, 36, 61, 50, 48, 45, 57],
            "K": [43, 41, 44, 40, 42, 38, 35, 30, 33, 45, 44, 42, 32, 46, 39,
                  47, 36, 41, 37, 43, 48, 40, 34, 44, 45, 38, 31, 42, 40, 46,
                  35, 41, 33, 39, 43, 49, 37, 32, 44, 41, 50, 38, 30, 43, 42],
            "temperature": [20, 25, 30, 22, 26, 28, 32, 34, 27, 23, 21, 29, 33, 24, 25,
                           22, 31, 23, 26, 28, 25, 30, 32, 24, 27, 29, 34, 22, 26, 21,
                           28, 25, 31, 23, 27, 24, 30, 32, 26, 25, 22, 29, 33, 27, 23],
            "humidity": [82, 80, 75, 85, 70, 65, 60, 55, 72, 78, 83, 68, 62, 77, 71,
                        79, 58, 86, 73, 67, 81, 64, 61, 76, 69, 84, 59, 80, 74, 87,
                        66, 75, 63, 85, 70, 82, 60, 57, 72, 79, 88, 69, 65, 78, 76],
            "ph": [6.5, 7.0, 6.8, 6.2, 6.9, 7.1, 6.3, 6.0, 6.7, 6.4, 6.6, 7.0, 6.1, 6.8, 6.5,
                   6.9, 6.4, 6.7, 6.6, 6.8, 7.1, 6.3, 6.2, 6.9, 6.5, 7.0, 6.0, 6.7, 6.4, 6.8,
                   6.5, 6.9, 6.2, 6.6, 7.0, 6.3, 6.7, 6.1, 6.8, 6.4, 7.2, 6.5, 6.3, 6.9, 6.6],
            "rainfall": [200, 180, 150, 210, 170, 160, 140, 120, 155, 190, 195, 165, 135, 205, 175,
                        185, 145, 215, 160, 168, 202, 152, 138, 198, 172, 192, 128, 188, 178, 208,
                        158, 182, 142, 212, 170, 198, 162, 132, 185, 177, 218, 167, 148, 200, 186],
            "crop": [
                # Rice (5 samples)
                "Rice", "Rice", "Rice", "Rice", "Rice",
                # Wheat (5 samples)
                "Wheat", "Wheat", "Wheat", "Wheat", "Wheat",
                # Maize (5 samples)
                "Maize", "Maize", "Maize", "Maize", "Maize",
                # Cotton (5 samples)
                "Cotton", "Cotton", "Cotton", "Cotton", "Cotton",
                # Sugarcane (5 samples)
                "Sugarcane", "Sugarcane", "Sugarcane", "Sugarcane", "Sugarcane",
                # Millet (5 samples)
                "Millet", "Millet", "Millet", "Millet", "Millet",
                # Pulses (5 samples)
                "Pulses", "Pulses", "Pulses", "Pulses", "Pulses",
                # Groundnut (5 samples)
                "Groundnut", "Groundnut", "Groundnut", "Groundnut", "Groundnut",
                # Soybean (5 samples)
                "Soybean", "Soybean", "Soybean", "Soybean", "Soybean",
            ]
        }
        
        self.training_data = pd.DataFrame(data)
        return self.training_data
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training"""
        
        X = self.training_data.drop("crop", axis=1)
        y = self.training_data["crop"]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_model(self, model_type="gradient_boost"):
        """Train the model"""
        
        if model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        else:
            raise ValueError("Unknown model type")
        
        self.model.fit(self.X_train_scaled, self.y_train)
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=5)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        evaluation = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
        
        return evaluation
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        importance = self.model.feature_importances_
        
        feature_importance_dict = dict(zip(feature_names, importance))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features
    
    def save_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        """Save trained model and scaler"""
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
    
    def load_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        """Load trained model and scaler"""
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Scaler loaded from {scaler_path}")

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Crop Recommendation System - Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Create training data
    print("\n📊 Creating training dataset...")
    trainer.create_training_data()
    print(f"✅ Dataset created with {len(trainer.training_data)} samples")
    print(f"   Crops: {trainer.training_data['crop'].unique().tolist()}")
    
    # Prepare data
    print("\n🔄 Preparing data...")
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\n🧠 Training Gradient Boosting model...")
    trainer.train_model("gradient_boost")
    print("✅ Model training completed")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    evaluation = trainer.evaluate_model()
    print(f"✅ Model Accuracy: {evaluation['accuracy']:.4f}")
    print(f"✅ Cross-validation Score: {evaluation['cv_mean']:.4f} (+/- {evaluation['cv_std']:.4f})")
    
    # Feature importance
    print("\n⭐ Feature Importance:")
    importance = trainer.get_feature_importance()
    for i, (feature, imp) in enumerate(importance, 1):
        print(f"   {i}. {feature}: {imp:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("✅ Training pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
