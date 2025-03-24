import os
import speech_recognition as sr
import pyttsx3
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import scrolledtext, messagebox
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import numpy as np
import re

class MediMindAssistant:
    def __init__(self):
        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 210)
        self.engine.setProperty('volume', 1)
        
        # PDF output path
        self.pdf_path = f"MediMind_Prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Medicine list
        self.medicines = []
        
        # Medical specialties for organizing medicines
        self.specialties = {
            'cardiology': [],
            'neurology': [],
            'respiratory': [],
            'gastroenterology': [],
            'endocrinology': [],
            'antibiotics': [],
            'analgesics': [],
            'other': []
        }
        
        # Load or train the model
        self.model = self.load_or_train_model()
        
        # Load medicines pronunciation dictionary
        self.pronunciation_dict = self.load_pronunciation_dict()

    def speak(self, text):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def load_pronunciation_dict(self):
        """Load or create a pronunciation dictionary for medicine names"""
        pronunciation_file = "medicine_pronunciations.pkl"
        if not os.path.exists(pronunciation_file):
            # Create pronunciation variations for common medicines
            pronunciations = {}
            # Get the medicines from our dataset
            medicines = self.get_medicine_dataset()['medicine_name'].unique()
            
            for med in medicines:
                # Create common pronunciation variations
                variations = [med.lower()]
                # Add variations with different spacing
                if ' ' in med:
                    variations.append(med.replace(' ', '').lower())
                # Add common mispronunciations
                med_lower = med.lower()
                if 'c' in med_lower:
                    variations.append(med_lower.replace('c', 'k'))
                if 'ph' in med_lower:
                    variations.append(med_lower.replace('ph', 'f'))
                if 'x' in med_lower:
                    variations.append(med_lower.replace('x', 'ks'))
                
                # Add all variations to dictionary
                for variation in variations:
                    pronunciations[variation] = med
            
            joblib.dump(pronunciations, pronunciation_file)
            return pronunciations
        else:
            return joblib.load(pronunciation_file)

    def get_medicine_dataset(self):
        """Create a comprehensive dataset of medicines"""
        # Create medicine data file path
        medicine_data_file = "medicine_data.csv"
        
        # If the file doesn't exist, create it
        if not os.path.exists(medicine_data_file):
            # Create a comprehensive dataset with medicine names, categories, and dosages
            data = {
                'medicine_name': [
                    # Cardiovascular medicines
                    'Atorvastatin', 'Simvastatin', 'Lisinopril', 'Amlodipine', 'Metoprolol',
                    'Atenolol', 'Clopidogrel', 'Warfarin', 'Aspirin', 'Furosemide',
                    'Losartan', 'Valsartan', 'Digoxin', 'Nitroglycerin', 'Diltiazem',
                    'Pravastatin', 'Carvedilol', 'Propranolol', 'Ramipril', 'Amiodarone',
                    
                    # Neurological medicines
                    'Levetiracetam', 'Gabapentin', 'Pregabalin', 'Lamotrigine', 'Escitalopram',
                    'Sertraline', 'Fluoxetine', 'Amitriptyline', 'Duloxetine', 'Valproic Acid',
                    'Alprazolam', 'Clonazepam', 'Lorazepam', 'Diazepam', 'Zolpidem',
                    'Memantine', 'Donepezil', 'Quetiapine', 'Risperidone', 'Olanzapine',
                    
                    # Respiratory medicines
                    'Salbutamol', 'Fluticasone', 'Budesonide', 'Montelukast', 'Tiotropium',
                    'Formoterol', 'Salmeterol', 'Ipratropium', 'Cetirizine', 'Loratadine',
                    'Fexofenadine', 'Fluticasone Propionate', 'Beclomethasone', 'Prednisolone', 'Prednisone',
                    'Dextromethorphan', 'Codeine', 'Guaifenesin', 'Pseudoephedrine', 'Phenylephrine',
                    
                    # Gastrointestinal medicines
                    'Omeprazole', 'Pantoprazole', 'Lansoprazole', 'Ranitidine', 'Famotidine',
                    'Ondansetron', 'Metoclopramide', 'Domperidone', 'Loperamide', 'Bisacodyl',
                    'Lactulose', 'Senna', 'Docusate', 'Mesalazine', 'Sulfasalazine',
                    'Esomeprazole', 'Rabeprazole', 'Sucralfate', 'Simethicone', 'Mebeverine',
                    
                    # Endocrine medicines
                    'Metformin', 'Gliclazide', 'Glimepiride', 'Sitagliptin', 'Empagliflozin',
                    'Dapagliflozin', 'Insulin Glargine', 'Insulin Aspart', 'Levothyroxine', 'Carbimazole',
                    'Propylthiouracil', 'Prednisolone', 'Hydrocortisone', 'Dexamethasone', 'Fludrocortisone',
                    'Alendronic Acid', 'Risedronate', 'Testosterone', 'Estradiol', 'Progesterone',
                    
                    # Antibiotics
                    'Amoxicillin', 'Flucloxacillin', 'Clarithromycin', 'Erythromycin', 'Azithromycin',
                    'Ciprofloxacin', 'Levofloxacin', 'Trimethoprim', 'Nitrofurantoin', 'Metronidazole',
                    'Doxycycline', 'Tetracycline', 'Cephalexin', 'Cefuroxime', 'Ceftriaxone',
                    'Meropenem', 'Gentamicin', 'Vancomycin', 'Clindamycin', 'Co-amoxiclav',
                    
                    # Analgesics
                    'Paracetamol', 'Ibuprofen', 'Naproxen', 'Diclofenac', 'Codeine Phosphate',
                    'Tramadol', 'Morphine', 'Oxycodone', 'Fentanyl', 'Buprenorphine',
                    'Aspirin', 'Celecoxib', 'Etoricoxib', 'Mefenamic Acid', 'Gabapentin',
                    'Pregabalin', 'Amitriptyline', 'Duloxetine', 'Nortriptyline', 'Co-codamol',
                    
                    # Common OTC medicines
                    'Acetaminophen', 'Ibuprofen', 'Diphenhydramine', 'Loratadine', 'Cetirizine',
                    'Fexofenadine', 'Ranitidine', 'Loperamide', 'Docusate', 'Bisacodyl',
                    'Senna', 'Glycerin', 'Hydrocortisone Cream', 'Clotrimazole', 'Miconazole',
                    'Menthol', 'Camphor', 'Benzocaine', 'Oxymetazoline', 'Zinc Oxide'
                ],
                'category': [
                    # Cardiovascular medicines
                    'cardiology', 'cardiology', 'cardiology', 'cardiology', 'cardiology',
                    'cardiology', 'cardiology', 'cardiology', 'cardiology', 'cardiology',
                    'cardiology', 'cardiology', 'cardiology', 'cardiology', 'cardiology',
                    'cardiology', 'cardiology', 'cardiology', 'cardiology', 'cardiology',
                    
                    # Neurological medicines
                    'neurology', 'neurology', 'neurology', 'neurology', 'neurology',
                    'neurology', 'neurology', 'neurology', 'neurology', 'neurology',
                    'neurology', 'neurology', 'neurology', 'neurology', 'neurology',
                    'neurology', 'neurology', 'neurology', 'neurology', 'neurology',
                    
                    # Respiratory medicines
                    'respiratory', 'respiratory', 'respiratory', 'respiratory', 'respiratory',
                    'respiratory', 'respiratory', 'respiratory', 'respiratory', 'respiratory',
                    'respiratory', 'respiratory', 'respiratory', 'respiratory', 'respiratory',
                    'respiratory', 'respiratory', 'respiratory', 'respiratory', 'respiratory',
                    
                    # Gastrointestinal medicines
                    'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology',
                    'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology',
                    'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology',
                    'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology', 'gastroenterology',
                    
                    # Endocrine medicines
                    'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology',
                    'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology',
                    'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology',
                    'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology', 'endocrinology',
                    
                    # Antibiotics
                    'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics',
                    'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics',
                    'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics',
                    'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics', 'antibiotics',
                    
                    # Analgesics
                    'analgesics', 'analgesics', 'analgesics', 'analgesics', 'analgesics',
                    'analgesics', 'analgesics', 'analgesics', 'analgesics', 'analgesics',
                    'analgesics', 'analgesics', 'analgesics', 'analgesics', 'analgesics',
                    'analgesics', 'analgesics', 'analgesics', 'analgesics', 'analgesics',
                    
                    # Common OTC medicines
                    'other', 'other', 'other', 'other', 'other',
                    'other', 'other', 'other', 'other', 'other',
                    'other', 'other', 'other', 'other', 'other',
                    'other', 'other', 'other', 'other', 'other'
                ],
                'common_dosage': [
                    # Cardiovascular dosages
                    '10-80mg once daily', '10-40mg once daily', '10-40mg once daily', '5-10mg once daily', '25-100mg twice daily',
                    '25-100mg once daily', '75mg once daily', '1-10mg as directed', '75-300mg once daily', '20-80mg once or twice daily',
                    '25-100mg once daily', '80-320mg once daily', '62.5-250mcg once daily', '300-600mcg as needed', '60-360mg daily',
                    '10-40mg once daily', '3.125-25mg twice daily', '40-320mg daily', '2.5-10mg once daily', '100-400mg daily',
                    
                    # Neurological dosages
                    '500-1500mg twice daily', '100-600mg three times daily', '150-600mg twice daily', '25-200mg twice daily', '10-20mg once daily',
                    '50-200mg once daily', '20-60mg once daily', '10-75mg once daily', '30-120mg once daily', '200-500mg twice daily',
                    '0.25-0.5mg three times daily', '0.5-2mg twice daily', '1-4mg daily', '2-10mg daily', '5-10mg once daily',
                    '5-20mg once daily', '5-10mg once daily', '25-800mg daily', '0.5-6mg daily', '2.5-20mg daily',
                    
                    # Respiratory dosages
                    '100-200mcg as needed', '100-250mcg twice daily', '200-400mcg twice daily', '10mg once daily', '18mcg once daily',
                    '12mcg twice daily', '50mcg twice daily', '20-40mcg 3-4 times daily', '10mg once daily', '10mg once daily',
                    '120-180mg once daily', '100-500mcg daily', '100-400mcg daily', '5-60mg daily', '5-60mg daily',
                    '15-30mg 3-4 times daily', '15-30mg 3-4 times daily', '200-400mg 4 times daily', '60mg 4 times daily', '10mg 4 times daily',
                    
                    # Gastrointestinal dosages
                    '20-40mg once daily', '20-80mg once daily', '15-30mg once daily', '150mg twice daily', '20-40mg once daily',
                    '4-8mg three times daily', '10mg three times daily', '10mg three times daily', '2-4mg as needed', '5-10mg once daily',
                    '15-30ml twice daily', '7.5-15mg once or twice daily', '100mg once or twice daily', '400mg three times daily', '1g three times daily',
                    '20-40mg once daily', '20mg once daily', '1g four times daily', '40-125mg as needed', '135-200mg three times daily',
                    
                    # Endocrine dosages
                    '500-1000mg twice daily', '40-320mg daily', '1-4mg once daily', '100mg once daily', '10-25mg once daily',
                    '5-10mg once daily', '10-50 units daily', 'As directed', '25-200mcg once daily', '10-60mg daily',
                    '50-300mg daily', '5-60mg daily', '20-300mg daily', '0.5-10mg daily', '50-300mcg daily',
                    '10mg once daily', '35mg once weekly', 'As directed', 'As directed', 'As directed',
                    
                    # Antibiotics dosages
                    '250-500mg three times daily', '250-500mg four times daily', '250-500mg twice daily', '250-500mg four times daily', '250-500mg once daily',
                    '250-750mg twice daily', '250-500mg once daily', '200mg twice daily', '50-100mg four times daily', '400mg three times daily',
                    '100mg daily', '250-500mg four times daily', '250-500mg four times daily', '250-500mg twice daily', '1-2g once daily',
                    '500mg-1g three times daily', '1-5mg/kg daily', '500mg-1g twice daily', '150-450mg four times daily', '250/125mg to 500/125mg three times daily',
                    
                    # Analgesics dosages
                    '500-1000mg 4-6 times daily', '200-400mg three times daily', '250-500mg twice daily', '50mg three times daily', '30-60mg four times daily',
                    '50-100mg four times daily', '10-30mg every 4 hours', '5-10mg every 4-6 hours', 'As directed', '200-400mcg daily',
                    '300-900mg four times daily', '100-200mg once daily', '30-120mg once daily', '500mg three times daily', '300-600mg three times daily',
                    '150-300mg twice daily', '10-25mg three times daily', '60mg once daily', '10-25mg three times daily', '8/500mg to 30/500mg as needed',
                    
                    # OTC dosages
                    '500-1000mg 4-6 times daily', '200-400mg three times daily', '25-50mg as needed', '10mg once daily', '10mg once daily',
                    '120-180mg once daily', '150mg twice daily', '2-4mg as needed', '100mg once or twice daily', '5-10mg once daily',
                    '7.5-15mg once or twice daily', 'As directed', 'Apply as needed', 'Apply as directed', 'Apply as directed',
                    'Apply as needed', 'Apply as needed', 'Apply as needed', 'As directed', 'Apply as needed'
                ]
            }
            
            # Create the DataFrame and save to CSV
            df = pd.DataFrame(data)
            
            # Augment dataset with common variations and misspellings
            augmented_rows = []
            for _, row in df.iterrows():
                # Add common brand names and spelling variations for some medicines
                medicine = row['medicine_name'].lower()
                
                # Simple rule-based misspellings and variations
                if medicine == 'paracetamol':
                    augmented_rows.append({
                        'medicine_name': 'Tylenol',
                        'category': row['category'],
                        'common_dosage': row['common_dosage']
                    })
                elif medicine == 'ibuprofen':
                    augmented_rows.append({
                        'medicine_name': 'Advil',
                        'category': row['category'],
                        'common_dosage': row['common_dosage']
                    })
                    augmented_rows.append({
                        'medicine_name': 'Nurofen',
                        'category': row['category'],
                        'common_dosage': row['common_dosage']
                    })
                elif medicine == 'acetaminophen':
                    augmented_rows.append({
                        'medicine_name': 'Paracetamol',
                        'category': row['category'],
                        'common_dosage': row['common_dosage']
                    })
                elif medicine == 'salbutamol':
                    augmented_rows.append({
                        'medicine_name': 'Ventolin',
                        'category': row['category'],
                        'common_dosage': row['common_dosage']
                    })
                    augmented_rows.append({
                        'medicine_name': 'Albuterol',
                        'category': row['category'],
                        'common_dosage': row['common_dosage']
                    })
            
            # Add the augmented rows to the dataset
            augmented_df = pd.DataFrame(augmented_rows)
            df = pd.concat([df, augmented_df], ignore_index=True)
            
            # Save to CSV
            df.to_csv(medicine_data_file, index=False)
            return df
        else:
            return pd.read_csv(medicine_data_file)

    def load_or_train_model(self):
        """Load or train a model for medicine name recognition with improved accuracy"""
        model_file = "medicine_model.pkl"
        model_metrics_file = "model_metrics.json"
        
        if not os.path.exists(model_file):
            print("Training new medicine recognition model...")
            # Get the comprehensive medicine dataset
            df = self.get_medicine_dataset()
            
            # Prepare data for training
            X = df['medicine_name'].str.lower()
            y = df['medicine_name']
            
            # Create training data with augmentations for better speech recognition
            training_data = []
            training_labels = []
            
            for medicine, label in zip(X, y):
                # Original form
                training_data.append(medicine)
                training_labels.append(label)
                
                # Common speech recognition errors and variations
                # Remove spaces
                if ' ' in medicine:
                    training_data.append(medicine.replace(' ', ''))
                    training_labels.append(label)
                
                # Replace ph with f (common pronunciation)
                if 'ph' in medicine:
                    training_data.append(medicine.replace('ph', 'f'))
                    training_labels.append(label)
                
                # Replace z with s (common pronunciation)
                if 'z' in medicine:
                    training_data.append(medicine.replace('z', 's'))
                    training_labels.append(label)
                    
                # Remove silent letters
                if medicine.endswith('e'):
                    training_data.append(medicine[:-1])
                    training_labels.append(label)
            
            # Convert to DataFrame
            train_df = pd.DataFrame({
                'text': training_data,
                'label': training_labels
            })
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                train_df['text'], train_df['label'], test_size=0.2, random_state=42
            )
            
            # Create and train the model with improved parameters
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',  # Character n-grams within word boundaries
                ngram_range=(2, 5),  # Use character n-grams from 2 to 5 characters
                max_features=5000,
                sublinear_tf=True,
                min_df=2
            )
            
            model = make_pipeline(
                vectorizer,
                MultinomialNB(alpha=0.1)  # Slightly lower alpha for less smoothing
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model metrics
            metrics = {
                "accuracy": accuracy,
                "num_medicines": len(df['medicine_name'].unique()),
                "training_examples": len(X_train),
                "test_examples": len(X_test)
            }
            
            # Print metrics
            print(f"Model trained with accuracy: {accuracy:.4f}")
            print(f"Number of medicines: {metrics['num_medicines']}")
            
            # Save model and metrics
            joblib.dump(model, model_file)
            with open(model_metrics_file, 'w') as f:
                import json
                json.dump(metrics, f)
            
            return model
        else:
            return joblib.load(model_file)

    def preprocess_command(self, command):
        """Preprocess the voice command to improve recognition"""
        if not command:
            return None
            
        # Convert to lowercase
        command = command.lower()
        
        # Remove common filler words and punctuation
        fillers = ['um', 'uh', 'like', 'so', 'and', 'then', 'also', 'please', 'add', 'prescribe']
        for filler in fillers:
            command = re.sub(r'\b' + filler + r'\b', ' ', command)
            
        # Remove extra whitespace
        command = re.sub(r'\s+', ' ', command).strip()
        
        # Replace common dosage patterns
        command = re.sub(r'\d+\s*mg', '', command)
        command = re.sub(r'\d+\s*mcg', '', command)
        command = re.sub(r'\d+\s*ml', '', command)
        
        # Remove common instructions
        command = re.sub(r'once daily', '', command)
        command = re.sub(r'twice daily', '', command)
        command = re.sub(r'three times daily', '', command)
        command = re.sub(r'four times daily', '', command)
        
        return command

    def classify_medicine(self, command):
        """Classify spoken command as a medicine name with improved accuracy"""
        if not command:
            return None
        
        # Clean the command
        processed_command = self.preprocess_command(command)
        if not processed_command:
            return None
            
        # Check if it's in our pronunciation dictionary first (exact match)
        if processed_command in self.pronunciation_dict:
            return self.pronunciation_dict[processed_command]
            
        # Try to predict using the trained model
        try:
            prediction = self.model.predict([processed_command])[0]
            confidence = np.max(self.model.steps[1][1].predict_proba(
                self.model.steps[0][1].transform([processed_command])
            ))
            
            # Only return prediction if confidence is high enough
            if confidence > 0.6:
                return prediction
            else:
                # For lower confidence matches, use similarity to find best match
                # Get medicines dataset
                medicines = self.get_medicine_dataset()['medicine_name'].unique()
                
                # Calculate string similarity with all medicines
                best_match = None
                best_score = 0
                
                for med in medicines:
                    # Simple Levenshtein distance ratio
                    score = self.string_similarity(processed_command, med.lower())
                    if score > best_score and score > 0.7:
                        best_score = score
                        best_match = med
                
                return best_match if best_match else command  # Return original if no good match
        except Exception as e:
            print(f"Error in classifying medicine: {e}")
            return command  # Return the original command if classification fails

    def string_similarity(self, s1, s2):
        """Calculate similarity between two strings (simple Levenshtein distance ratio)"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()

    def add_medicine_with_details(self, medicine_name, dosage=None, frequency=None):
        """Add medicine with additional details like dosage and frequency"""
        # Get default dosage from our dataset if available
        medicine_data = self.get_medicine_dataset()
        default_dosage = None
        
        # Look up the medicine in our dataset
        medicine_row = medicine_data[medicine_data['medicine_name'].str.lower() == medicine_name.lower()]
        
        if not medicine_row.empty:
            default_dosage = medicine_row.iloc[0]['common_dosage']
            category = medicine_row.iloc[0]['category']
            
            # Add to appropriate specialty list
            if category in self.specialties:
                self.specialties[category].append(medicine_name)
        else:
            # If not found, add to "other" category
            self.specialties['other'].append(medicine_name)
        
        # Add to general medicines list with dosage and frequency
        medicine_info = {
            'name': medicine_name,
            'dosage': dosage if dosage else default_dosage,
            'frequency': frequency if frequency else 'As directed'
        }
        
        self.medicines.append(medicine_info)
        return medicine_info

    def take_command(self):
        """Capture voice command with improved noise handling"""
        try:
            with sr.Microphone() as source:
                print("Listening for medicines...")
                # Adjust for ambient noise with longer duration for better adaptation
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                # Allow more time for the doctor to speak and increase phrase time limit
                audio = self.recognizer.listen(source, timeout=20, phrase_time_limit=15)
                command = self.recognizer.recognize_google(audio)
                print(f"Doctor said: {command}")
                return command.lower()
        except sr.UnknownValueError:
            self.speak("Sorry, I didn't understand that")
            return None
        except sr.RequestError as e:
            self.speak(f"Speech service error: {e}")
            return None
        except Exception as e:
            self.speak(f"An error occurred: {e}")
            return None

    def extract_dosage_info(self, command):
        """Extract dosage and frequency from command"""
        # Define patterns for common dosage formats
        dosage_patterns = [
            r'(\d+)\s*mg',  # 10mg, 20 mg, etc.
            r'(\d+)\s*mcg', # 100mcg, 50 mcg, etc.
            r'(\d+)\s*ml',  # 5ml, 10 ml, etc.
            r'(\d+)\s*g',   # 1g, 2 g, etc.
            r'(\d+\.\d+)\s*mg', # 2.5mg, 7.5 mg, etc.
            r'(\d+)/(\d+)',  # 10/500, 5/325, etc. (combination medicines)
        ]
        
        # Define patterns for common frequency formats
        frequency_patterns = [
            r'once daily',
            r'twice daily',
            r'three times daily',
            r'(\d+) times daily',
            r'every (\d+) hours',
            r'as needed',
            r'before meals',
            r'after meals',
            r'at bedtime',
            r'in the morning',
        ]
        
        # Extract dosage
        dosage = None
        for pattern in dosage_patterns:
            match = re.search(pattern, command)
            if match:
                dosage = match.group(0)
                break
        
        # Extract frequency
        frequency = None
        for pattern in frequency_patterns:
            match = re.search(pattern, command)
            if match:
                frequency = match.group(0)
                break
        
        return dosage, frequency

    def generate_prescription_pdf(self):
        """Generate a PDF prescription with organized medicines by specialty"""
        # Create PDF instance
        pdf = FPDF()
        pdf.add_page()
        
        # Set up header
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "MediMind Prescription", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
        pdf.ln(10)
        
        # Add medicines organized by specialty
        pdf.set_font("Arial", "B", 14)
        
        # Loop through specialties
        for specialty, medicines_list in self.specialties.items():
            if medicines_list:  # Only include specialties that have medicines
                # Capitalize first letter of specialty
                pdf.cell(0, 10, f"{specialty.capitalize()} Medicines", ln=True)
                pdf.set_font("Arial", "", 12)
                
                # Add each medicine from this specialty with its details
                for medicine_name in medicines_list:
                    # Find full details from our main medicines list
                    medicine_details = next((med for med in self.medicines if med['name'] == medicine_name), None)
                    
                    if medicine_details:
                        # Format medicine information with asterisk instead of bullet
                        medicine_text = f"* {medicine_details['name']}"
                        if medicine_details['dosage']:
                            medicine_text += f" - {medicine_details['dosage']}"
                        if medicine_details['frequency']:
                            medicine_text += f" - {medicine_details['frequency']}"
                        
                        pdf.multi_cell(0, 10, medicine_text)
                
                pdf.ln(5)
                pdf.set_font("Arial", "B", 14)
        
        # Add footer with signature line
        pdf.ln(10)
        pdf.line(20, pdf.get_y(), 90, pdf.get_y())
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "Doctor's Signature", ln=True)
        
        # Save PDF
        pdf.output(self.pdf_path)
        return self.pdf_path
    def create_gui(self):
        """Create a GUI for the MediMind Assistant"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("MediMind Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Create header
        header_frame = tk.Frame(self.root, bg="#3498db", padx=10, pady=10)
        header_frame.pack(fill="x")
        
        tk.Label(
            header_frame, 
            text="MediMind Voice Assistant", 
            font=("Arial", 18, "bold"), 
            fg="white",
            bg="#3498db"
        ).pack()
        
        # Create main content frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Create medicine list display
        list_frame = tk.Frame(main_frame, bg="#f0f0f0")
        list_frame.pack(fill="both", expand=True, pady=10)
        
        tk.Label(
            list_frame, 
            text="Prescribed Medicines:", 
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        ).pack(anchor="w")
        
        self.medicine_display = scrolledtext.ScrolledText(list_frame, width=80, height=20)
        self.medicine_display.pack(fill="both", expand=True)
        
        # Create button frame
        button_frame = tk.Frame(main_frame, bg="#f0f0f0", pady=10)
        button_frame.pack(fill="x")
        
        # Add medicine button
        add_button = tk.Button(
            button_frame,
            text="Add Medicine (Voice)",
            command=self.add_medicine_voice,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5
        )
        add_button.pack(side="left", padx=5)
        
        # Generate PDF button
        pdf_button = tk.Button(
            button_frame,
            text="Generate Prescription PDF",
            command=self.generate_pdf_action,
            bg="#3498db",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5
        )
        pdf_button.pack(side="left", padx=5)
        
        # Clear all button
        clear_button = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_all,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5
        )
        clear_button.pack(side="left", padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update the display
        self.update_medicine_display()
        
        # Start the GUI
        self.root.mainloop()

    def add_medicine_voice(self):
        """Capture voice and add medicine to prescription"""
        self.status_var.set("Listening for medicine...")
        self.root.update_idletasks()
        
        # Get voice command
        command = self.take_command()
        
        if command:
            # Extract medicine name and details
            medicine_name = self.classify_medicine(command)
            
            if medicine_name:
                # Extract dosage and frequency if present in the command
                dosage, frequency = self.extract_dosage_info(command)
                
                # Add medicine with details
                med_info = self.add_medicine_with_details(medicine_name, dosage, frequency)
                
                # Update GUI
                self.update_medicine_display()
                
                # Feedback
                self.status_var.set(f"Added: {med_info['name']} - {med_info['dosage']} - {med_info['frequency']}")
                self.speak(f"Added {med_info['name']}")
            else:
                self.status_var.set("Could not recognize medicine name")
                self.speak("I couldn't recognize the medicine name")
        else:
            self.status_var.set("No command detected")

    def update_medicine_display(self):
        """Update the medicine display in the GUI"""
        # Clear current display
        self.medicine_display.delete(1.0, tk.END)
        
        # No medicines added yet
        if not self.medicines:
            self.medicine_display.insert(tk.END, "No medicines added yet. Click 'Add Medicine' to begin.")
            return
        
        # Display medicines by specialty
        for specialty, medicines_list in self.specialties.items():
            if medicines_list:
                # Add specialty header
                self.medicine_display.insert(tk.END, f"\n{specialty.upper()}\n", "specialty")
                self.medicine_display.tag_configure("specialty", font=("Arial", 12, "bold"))
                
                # Add medicines in this specialty
                for medicine_name in medicines_list:
                    # Find full details
                    medicine_details = next((med for med in self.medicines if med['name'] == medicine_name), None)
                    
                    if medicine_details:
                        medicine_text = f"• {medicine_details['name']}"
                        if medicine_details['dosage']:
                            medicine_text += f" - {medicine_details['dosage']}"
                        if medicine_details['frequency']:
                            medicine_text += f" - {medicine_details['frequency']}"
                        
                        self.medicine_display.insert(tk.END, f"{medicine_text}\n")

    def generate_pdf_action(self):
        """Generate PDF from GUI button"""
        if not self.medicines:
            messagebox.showinfo("MediMind", "No medicines added yet")
            return
            
        try:
            pdf_path = self.generate_prescription_pdf()
            self.status_var.set(f"PDF generated: {pdf_path}")
            self.speak("Prescription PDF has been generated")
            messagebox.showinfo("MediMind", f"Prescription saved to: {pdf_path}")
        except Exception as e:
            self.status_var.set(f"Error generating PDF: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}")

    def clear_all(self):
        """Clear all medicines from the prescription"""
        if messagebox.askyesno("MediMind", "Are you sure you want to clear all medicines?"):
            # Reset all data
            self.medicines = []
            self.specialties = {key: [] for key in self.specialties}
            
            # Update display
            self.update_medicine_display()
            self.status_var.set("All medicines cleared")

def main():
    # Create an instance of MediMindAssistant
    assistant = MediMindAssistant()
    
    # Welcome message
    assistant.speak("Welcome to MediMind Assistant")
    
    # Launch the GUI
    assistant.create_gui()

if __name__ == "__main__":
    main()