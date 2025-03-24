# # import pandas as pd
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
# # from sklearn.model_selection import train_test_split
# # import re
# # from tqdm import tqdm

# # # Configuration
# # MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# # MAX_LENGTH = 512
# # BATCH_SIZE = 8
# # LEARNING_RATE = 2e-5
# # EPOCHS = 3
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # class MedMCQADataset(Dataset):
# #     def __init__(self, data, tokenizer, max_length=512):
# #         self.data = data
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length
        
# #     def __len__(self):
# #         return len(self.data)
    
# #     def __getitem__(self, idx):
# #         item = self.data.iloc[idx]
        
# #         question = item['question']
# #         symptoms = item.get('symptoms', '')  # Get symptoms if available
        
# #         # Combine question and symptoms if available
# #         if symptoms and not pd.isna(symptoms):
# #             context = f"Question: {question}\nSymptoms: {symptoms}"
# #         else:
# #             context = f"Question: {question}"
            
# #         options = [
# #             item['opa'],
# #             item['opb'],
# #             item['opc'],
# #             item['opd']
# #         ]
        
# #         # Tokenize each option with the context
# #         encodings = []
# #         for option in options:
# #             encoding = self.tokenizer(
# #                 context,
# #                 option,
# #                 max_length=self.max_length,
# #                 padding="max_length",
# #                 truncation=True,
# #                 return_tensors="pt"
# #             )
# #             encodings.append({
# #                 "input_ids": encoding["input_ids"].squeeze(),
# #                 "attention_mask": encoding["attention_mask"].squeeze(),
# #                 "token_type_ids": encoding["token_type_ids"].squeeze() if "token_type_ids" in encoding else None
# #             })
        
# #         # Convert correct answer to index (1-based to 0-based)
# #         label = ord(item['cop'].lower()) - ord('a')
        
# #         # Stack all encodings
# #         input_ids = torch.stack([encoding["input_ids"] for encoding in encodings])
# #         attention_mask = torch.stack([encoding["attention_mask"] for encoding in encodings])
        
# #         if "token_type_ids" in encodings[0] and encodings[0]["token_type_ids"] is not None:
# #             token_type_ids = torch.stack([encoding["token_type_ids"] for encoding in encodings])
# #             return {
# #                 "input_ids": input_ids,
# #                 "attention_mask": attention_mask,
# #                 "token_type_ids": token_type_ids,
# #                 "labels": torch.tensor(label)
# #             }
# #         else:
# #             return {
# #                 "input_ids": input_ids,
# #                 "attention_mask": attention_mask,
# #                 "labels": torch.tensor(label)
# #             }

# # def preprocess_data(data_path):
# #     """Load and preprocess the MedMCQA dataset"""
# #     df = pd.read_csv(data_path)
    
# #     # Ensure all required columns exist
# #     required_cols = ['question', 'opa', 'opb', 'opc', 'opd', 'cop']
# #     if not all(col in df.columns for col in required_cols):
# #         missing = [col for col in required_cols if col not in df.columns]
# #         raise ValueError(f"Missing required columns: {missing}")
    
# #     # Check if symptoms column exists
# #     if 'symptoms' not in df.columns:
# #         print("Warning: 'symptoms' column not found in the dataset.")
    
# #     # Clean the data
# #     for col in df.columns:
# #         if df[col].dtype == 'object':
# #             df[col] = df[col].fillna('').astype(str)
    
# #     return df

# # def train_model(train_data, val_data, model_name, output_dir="./model_output"):
# #     """Train the medical QA model"""
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
# #     train_dataset = MedMCQADataset(train_data, tokenizer, MAX_LENGTH)
# #     val_dataset = MedMCQADataset(val_data, tokenizer, MAX_LENGTH)
    
# #     training_args = TrainingArguments(
# #         output_dir=output_dir,
# #         num_train_epochs=EPOCHS,
# #         per_device_train_batch_size=BATCH_SIZE,
# #         per_device_eval_batch_size=BATCH_SIZE,
# #         warmup_steps=500,
# #         weight_decay=0.01,
# #         logging_dir='./logs',
# #         logging_steps=100,
# #         eval_steps=500,
# #         save_steps=1000,
# #         evaluation_strategy="steps",
# #         load_best_model_at_end=True,
# #         learning_rate=LEARNING_RATE,
# #     )
    
# #     trainer = Trainer(
# #         model=model,
# #         args=training_args,
# #         train_dataset=train_dataset,
# #         eval_dataset=val_dataset
# #     )
    
# #     trainer.train()
    
# #     # Save the model and tokenizer
# #     model.save_pretrained(output_dir)
# #     tokenizer.save_pretrained(output_dir)
    
# #     return model, tokenizer

# # class MedicalChatbot:
# #     def __init__(self, model_path):
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
# #         self.model = AutoModelForMultipleChoice.from_pretrained(model_path)
# #         self.model.to(DEVICE)
# #         self.model.eval()
        
# #         # Load medical knowledge base if needed
# #         # self.knowledge_base = load_knowledge_base()
    
# #     def preprocess_query(self, question, symptoms=None):
# #         """Preprocess user query and symptoms"""
# #         if symptoms:
# #             context = f"Question: {question}\nSymptoms: {symptoms}"
# #         else:
# #             context = f"Question: {question}"
# #         return context
    
# #     def generate_options(self, question):
# #         """
# #         In a real system, you might use a model to generate plausible medical answer options
# #         For demo purposes, we'll use placeholders
# #         """
# #         return [
# #             "This is option A",
# #             "This is option B",
# #             "This is option C",
# #             "This is option D"
# #         ]
    
# #     def answer_question(self, question, symptoms=None, options=None):
# #         """Answer medical questions based on user input and symptoms"""
# #         context = self.preprocess_query(question, symptoms)
        
# #         # If no options provided, generate them (for demo)
# #         if not options:
# #             options = self.generate_options(question)
        
# #         # Prepare inputs for the model
# #         encodings = []
# #         for option in options:
# #             encoding = self.tokenizer(
# #                 context,
# #                 option,
# #                 max_length=MAX_LENGTH,
# #                 padding="max_length",
# #                 truncation=True,
# #                 return_tensors="pt"
# #             )
# #             encodings.append({
# #                 "input_ids": encoding["input_ids"].to(DEVICE),
# #                 "attention_mask": encoding["attention_mask"].to(DEVICE),
# #                 "token_type_ids": encoding["token_type_ids"].to(DEVICE) if "token_type_ids" in encoding else None
# #             })
        
# #         input_ids = torch.stack([encoding["input_ids"] for encoding in encodings]).squeeze(1)
# #         attention_mask = torch.stack([encoding["attention_mask"] for encoding in encodings]).squeeze(1)
        
# #         inputs = {
# #             "input_ids": input_ids,
# #             "attention_mask": attention_mask,
# #         }
        
# #         if "token_type_ids" in encodings[0] and encodings[0]["token_type_ids"] is not None:
# #             inputs["token_type_ids"] = torch.stack([encoding["token_type_ids"] for encoding in encodings]).squeeze(1)
        
# #         with torch.no_grad():
# #             outputs = self.model(**inputs)
# #             probs = torch.softmax(outputs.logits, dim=1)
# #             pred_idx = torch.argmax(probs, dim=1).item()
        
# #         confidence = probs[0][pred_idx].item()
# #         answer = options[pred_idx]
        
# #         # Format the response
# #         response = {
# #             "answer": answer,
# #             "confidence": f"{confidence:.2%}",
# #             "options": options,
# #             "selected_option": chr(ord('A') + pred_idx)
# #         }
        
# #         return response
    
# #     def format_response(self, response_data):
# #         """Format the model's response in a user-friendly way"""
# #         answer = response_data["answer"]
# #         confidence = response_data["confidence"]
        
# #         formatted_response = f"Based on the provided information, I believe the answer is:\n\n"
# #         formatted_response += f"**{answer}**\n\n"
# #         formatted_response += f"Confidence: {confidence}\n\n"
        
# #         if confidence < 0.7:
# #             formatted_response += "Note: This answer has lower confidence. Please consult with a healthcare professional for accurate medical advice."
        
# #         return formatted_response

# # # Main execution pipeline
# # def main():
# #     # Step 1: Load and preprocess data
# #     print("Loading and preprocessing data...")
# #     data_path = "path_to_medmcqa_with_symptoms.csv"  # Update with your file path
# #     df = preprocess_data(data_path)
    
# #     # Step 2: Split data
# #     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# #     print(f"Training data: {len(train_df)}, Validation data: {len(val_df)}")
    
# #     # Step 3: Train model
# #     print(f"Training model using {MODEL_NAME}...")
# #     model, tokenizer = train_model(train_df, val_df, MODEL_NAME)
# #     print("Model training completed!")
    
# #     # Step 4: Initialize chatbot
# #     chatbot = MedicalChatbot("./model_output")
    
# #     # Step 5: Demo interface
# #     print("\n--- Medical QA System ---")
# #     print("Type 'exit' to quit the demo")
    
# #     while True:
# #         question = input("\nEnter your medical question: ")
# #         if question.lower() == 'exit':
# #             break
            
# #         symptoms = input("Enter any symptoms (optional): ")
# #         if not symptoms:
# #             symptoms = None
            
# #         # Process the query
# #         response_data = chatbot.answer_question(question, symptoms)
# #         formatted_response = chatbot.format_response(response_data)
        
# #         print("\n" + formatted_response)

# # if __name__ == "__main__":
# #     main()
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
# from sklearn.model_selection import train_test_split
# import re
# from tqdm import tqdm
# import random
# from faker import Faker
# import os
# import argparse

# # Initialize Faker for generating realistic-looking data
# fake = Faker()

# #########################
# # DATASET GENERATOR CODE
# #########################

# # Define categories of medical questions
# CATEGORIES = [
#     "Anatomy", "Physiology", "Biochemistry", "Pathology", "Microbiology", 
#     "Pharmacology", "Medicine", "Surgery", "Pediatrics", "Obstetrics", 
#     "Gynecology", "Psychiatry", "Dermatology", "ENT", "Ophthalmology", 
#     "Orthopedics", "Radiology", "Anesthesiology", "Emergency Medicine",
#     "Infectious Disease", "Cardiology", "Neurology", "Oncology", "Endocrinology"
# ]

# # Common symptoms by body system
# SYMPTOMS_BY_SYSTEM = {
#     "Cardiovascular": [
#         "chest pain", "palpitations", "shortness of breath", "dizziness", "fainting", 
#         "fatigue", "edema in legs", "cyanosis", "cold extremities", "hypertension"
#     ],
#     "Respiratory": [
#         "cough", "wheezing", "shortness of breath", "sputum production", "hemoptysis", 
#         "chest pain", "nasal congestion", "sore throat", "hoarseness", "sleep apnea"
#     ],
#     "Gastrointestinal": [
#         "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation", 
#         "blood in stool", "jaundice", "heartburn", "bloating", "loss of appetite",
#         "difficulty swallowing", "excessive gas", "dark urine", "pale stool"
#     ],
#     "Neurological": [
#         "headache", "dizziness", "numbness", "tingling", "tremor", 
#         "seizures", "confusion", "memory loss", "slurred speech", "difficulty walking",
#         "loss of consciousness", "weakness", "visual disturbances", "balance problems"
#     ],
#     "Musculoskeletal": [
#         "joint pain", "muscle pain", "stiffness", "swelling", "reduced range of motion", 
#         "back pain", "neck pain", "weakness", "bone pain", "muscle cramps"
#     ],
#     "Dermatological": [
#         "rash", "itching", "hives", "discoloration", "dryness", 
#         "lesions", "blisters", "bruising", "excessive sweating", "hair loss"
#     ],
#     "Endocrine": [
#         "fatigue", "weight changes", "excessive thirst", "frequent urination", "temperature intolerance", 
#         "increased hunger", "hair changes", "mood changes", "sexual dysfunction", "slow wound healing"
#     ],
#     "Psychiatric": [
#         "anxiety", "depression", "mood swings", "sleep disturbances", "hallucinations", 
#         "delusions", "irritability", "social withdrawal", "concentration issues", "suicidal thoughts"
#     ],
#     "Urological": [
#         "frequent urination", "painful urination", "blood in urine", "incontinence", "urgency", 
#         "hesitancy", "decreased urine output", "nocturia", "flank pain", "genital pain"
#     ],
#     "Immunological": [
#         "frequent infections", "prolonged wound healing", "fatigue", "joint pain", "rashes", 
#         "low-grade fever", "swollen lymph nodes", "allergic reactions", "autoimmune symptoms"
#     ]
# }

# # Question templates for creating medical MCQs
# QUESTION_TEMPLATES = [
#     "What is the most likely diagnosis for a patient presenting with {symptoms}?",
#     "A {age}-year-old {gender} presents with {symptoms}. What is the most appropriate next step in management?",
#     "Which of the following is the most common cause of {condition}?",
#     "What is the mechanism of action of {drug}?",
#     "A patient with {condition} develops {symptoms}. What is the most appropriate treatment?",
#     "Which laboratory finding is most consistent with {condition}?",
#     "What is the most common complication of {condition}?",
#     "Which imaging study is most appropriate for a patient with {symptoms}?",
#     "A {age}-year-old {gender} presents with {symptoms} for {duration}. Which of the following is the most likely diagnosis?",
#     "Which of the following is a contraindication for {treatment} in patients with {condition}?",
#     "What is the gold standard diagnostic test for {condition}?",
#     "Which medication is most appropriate for a patient with {condition} who presents with {symptoms}?",
#     "What is the characteristic pathological finding in {condition}?",
#     "Which risk factor is most strongly associated with {condition}?",
#     "What is the underlying pathophysiology of {symptoms} in patients with {condition}?",
#     "Which of the following best explains the {symptoms} experienced by patients with {condition}?",
#     "A patient with {condition} is started on {drug}. Which adverse effect should be monitored?",
#     "What is the most appropriate preventive measure for {condition}?",
#     "A {age}-year-old {gender} with a history of {condition} presents with {symptoms}. What is the most likely diagnosis?",
#     "Which anatomical structure is primarily affected in {condition}?",
#     "What is the function of {anatomical_structure}?",
#     "Which biochemical pathway is inhibited by {drug}?",
#     "A patient with {condition} should avoid which of the following?",
#     "What is the primary neurotransmitter involved in {condition}?",
#     "Which microorganism is the most common cause of {condition}?"
# ]

# # Medical conditions, drugs, tests, and treatments for generating realistic questions
# MEDICAL_DATA = {
#     "conditions": [
#         "hypertension", "diabetes mellitus", "asthma", "COPD", "myocardial infarction", 
#         "pneumonia", "appendicitis", "cholecystitis", "ulcerative colitis", "Crohn's disease",
#         "rheumatoid arthritis", "osteoarthritis", "multiple sclerosis", "Parkinson's disease", 
#         "Alzheimer's disease", "migraine", "epilepsy", "hypothyroidism", "hyperthyroidism", 
#         "Cushing's syndrome", "Addison's disease", "cirrhosis", "hepatitis", "gastric ulcer", 
#         "pulmonary embolism", "deep vein thrombosis", "osteoporosis", "gout", "psoriasis", 
#         "eczema", "systemic lupus erythematosus", "anemia", "leukemia", "lymphoma", 
#         "breast cancer", "prostate cancer", "lung cancer", "colorectal cancer", "cervical cancer", 
#         "urinary tract infection", "pyelonephritis", "renal calculi", "benign prostatic hyperplasia", 
#         "glaucoma", "cataracts", "macular degeneration", "otitis media", "sinusitis", 
#         "gastroesophageal reflux disease", "peptic ulcer disease", "pancreatitis", "gallstones"
#     ],
#     "drugs": [
#         "atorvastatin", "lisinopril", "metformin", "levothyroxine", "amlodipine", 
#         "albuterol", "omeprazole", "metoprolol", "losartan", "gabapentin", 
#         "sertraline", "fluoxetine", "escitalopram", "alprazolam", "zolpidem", 
#         "hydrochlorothiazide", "furosemide", "prednisone", "amoxicillin", "azithromycin", 
#         "ciprofloxacin", "acetaminophen", "ibuprofen", "naproxen", "aspirin", 
#         "warfarin", "clopidogrel", "apixaban", "insulin glargine", "insulin lispro", 
#         "metronidazole", "fluconazole", "hydroxychloroquine", "tamsulosin", "finasteride",
#         "sildenafil", "montelukast", "fluticasone", "budesonide", "methylprednisolone",
#         "diltiazem", "verapamil", "carvedilol", "digoxin", "amiodarone",
#         "phenytoin", "lamotrigine", "valproic acid", "levetiracetam", "topiramate"
#     ],
#     "tests": [
#         "complete blood count", "comprehensive metabolic panel", "lipid panel", "thyroid function tests", 
#         "hemoglobin A1c", "urinalysis", "liver function tests", "cardiac enzymes", "D-dimer", 
#         "electrocardiogram", "chest X-ray", "CT scan", "MRI", "ultrasound", "PET scan", 
#         "echocardiogram", "stress test", "colonoscopy", "endoscopy", "bronchoscopy", 
#         "bone density scan", "mammogram", "Pap smear", "prostate-specific antigen", "biopsy",
#         "electromyography", "electroencephalogram", "pulmonary function tests", "sleep study", "lumbar puncture",
#         "arteriogram", "venogram", "sputum culture", "blood culture", "urine culture"
#     ],
#     "treatments": [
#         "antibiotics", "antivirals", "antifungals", "antihypertensives", "antidiabetics", 
#         "bronchodilators", "corticosteroids", "anticoagulants", "antiplatelets", "statins", 
#         "proton pump inhibitors", "H2 blockers", "antidepressants", "anxiolytics", "antipsychotics", 
#         "anticonvulsants", "analgesics", "NSAIDs", "opioids", "immunosuppressants", 
#         "chemotherapy", "radiation therapy", "surgery", "dialysis", "transplantation", 
#         "physical therapy", "occupational therapy", "speech therapy", "cognitive behavioral therapy", "psychotherapy",
#         "diet modification", "exercise regimen", "weight loss", "smoking cessation", "alcohol cessation",
#         "oxygen therapy", "ventilation", "cardiac catheterization", "angioplasty", "bypass surgery"
#     ],
#     "anatomical_structures": [
#         "heart", "lungs", "liver", "kidneys", "brain", 
#         "spinal cord", "stomach", "intestines", "pancreas", "gallbladder", 
#         "thyroid", "adrenal glands", "pituitary gland", "testes", "ovaries", 
#         "esophagus", "trachea", "bronchi", "alveoli", "arteries", 
#         "veins", "capillaries", "lymph nodes", "spleen", "bone marrow", 
#         "joints", "muscles", "tendons", "ligaments", "cartilage", 
#         "skin", "hair", "nails", "eyes", "ears", 
#         "nose", "mouth", "tongue", "pharynx", "larynx"
#     ]
# }

# def generate_random_symptoms(num_symptoms=3):
#     """Generate a random set of symptoms"""
#     system = random.choice(list(SYMPTOMS_BY_SYSTEM.keys()))
#     symptoms = random.sample(SYMPTOMS_BY_SYSTEM[system], min(num_symptoms, len(SYMPTOMS_BY_SYSTEM[system])))
#     return ", ".join(symptoms)

# def generate_question_with_options():
#     """Generate a medical question with 4 options and one correct answer"""
#     category = random.choice(CATEGORIES)
    
#     # Select a question template
#     template = random.choice(QUESTION_TEMPLATES)
    
#     # Fill in the template with relevant information
#     age = random.randint(18, 85)
#     gender = random.choice(["male", "female"])
#     condition = random.choice(MEDICAL_DATA["conditions"])
#     drug = random.choice(MEDICAL_DATA["drugs"])
#     treatment = random.choice(MEDICAL_DATA["treatments"])
#     anatomical_structure = random.choice(MEDICAL_DATA["anatomical_structures"])
#     duration = f"{random.randint(1, 30)} {random.choice(['days', 'weeks', 'months', 'years'])}"
#     symptoms = generate_random_symptoms()
    
#     # Replace placeholders in the template
#     question = template.format(
#         age=age,
#         gender=gender,
#         condition=condition,
#         drug=drug,
#         treatment=treatment,
#         anatomical_structure=anatomical_structure,
#         duration=duration,
#         symptoms=symptoms
#     )
    
#     # Generate options based on the question type
#     if "diagnosis" in question.lower():
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
#     elif "treatment" in question.lower() or "management" in question.lower():
#         options = random.sample(MEDICAL_DATA["treatments"], 4)
#     elif "drug" in question.lower() or "medication" in question.lower():
#         options = random.sample(MEDICAL_DATA["drugs"], 4)
#     elif "test" in question.lower() or "diagnostic" in question.lower() or "imaging" in question.lower():
#         options = random.sample(MEDICAL_DATA["tests"], 4)
#     else:
#         # Mix of different options for other types of questions
#         all_options = (MEDICAL_DATA["conditions"] + MEDICAL_DATA["drugs"] + 
#                       MEDICAL_DATA["treatments"] + MEDICAL_DATA["anatomical_structures"])
#         options = random.sample(all_options, 4)
    
#     # Randomly select one option as the correct answer
#     correct_answer = random.choice(["A", "B", "C", "D"])
    
#     # Get separate symptoms for the symptoms field (may be different from those in the question)
#     separate_symptoms = generate_random_symptoms(num_symptoms=random.randint(2, 5))
    
#     return {
#         "category": category,
#         "question": question,
#         "opa": options[0],
#         "opb": options[1],
#         "opc": options[2],
#         "opd": options[3],
#         "cop": correct_answer,
#         "symptoms": separate_symptoms
#     }

# def generate_medmcqa_dataset(num_records=1000, output_file="synthetic_medmcqa_with_symptoms.csv"):
#     """Generate a synthetic MedMCQA dataset with symptoms"""
#     print(f"Generating {num_records} synthetic medical MCQs...")
    
#     data = []
#     for i in range(num_records):
#         record = generate_question_with_options()
#         # Add an ID field
#         record["id"] = f"SYNTHETIC_{i+1:05d}"
#         data.append(record)
        
#         # Print progress
#         if (i+1) % 100 == 0:
#             print(f"Generated {i+1} records...")
    
#     # Convert to DataFrame and save to CSV
#     df = pd.DataFrame(data)
    
#     # Reorder columns to match expected MedMCQA format with added symptoms
#     columns = ["id", "category", "question", "opa", "opb", "opc", "opd", "cop", "symptoms"]
#     df = df[columns]
    
#     df.to_csv(output_file, index=False)
#     print(f"Dataset saved to {output_file}")
    
#     return df

# def add_additional_fields(df):
#     """Add additional fields that might be useful for training"""
#     # Add difficulty level
#     df["difficulty"] = np.random.choice(["easy", "medium", "hard"], size=len(df))
    
#     # Add subject tags (multiple per question)
#     subjects = []
#     for _ in range(len(df)):
#         num_subjects = random.randint(1, 3)
#         subject_tags = random.sample(CATEGORIES, num_subjects)
#         subjects.append("|".join(subject_tags))
#     df["subject_tags"] = subjects
    
#     # Add explanation field
#     df["explanation"] = df.apply(
#         lambda row: fake.paragraph(nb_sentences=random.randint(3, 6)) + 
#                     f" The correct answer is {row['cop']}: {row[f'op{row['cop'].lower()}']}.",
#         axis=1
#     )
    
#     return df

# #########################
# # MODEL CODE
# #########################

# # Configuration
# MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# MAX_LENGTH = 512
# BATCH_SIZE = 8
# LEARNING_RATE = 2e-5
# EPOCHS = 3
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class MedMCQADataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data.iloc[idx]
        
#         question = item['question']
#         symptoms = item.get('symptoms', '')  # Get symptoms if available
        
#         # Combine question and symptoms if available
#         if symptoms and not pd.isna(symptoms):
#             context = f"Question: {question}\nSymptoms: {symptoms}"
#         else:
#             context = f"Question: {question}"
            
#         options = [
#             item['opa'],
#             item['opb'],
#             item['opc'],
#             item['opd']
#         ]
        
#         # Tokenize each option with the context
#         encodings = []
#         for option in options:
#             encoding = self.tokenizer(
#                 context,
#                 option,
#                 max_length=self.max_length,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             )
#             encodings.append({
#                 "input_ids": encoding["input_ids"].squeeze(),
#                 "attention_mask": encoding["attention_mask"].squeeze(),
#                 "token_type_ids": encoding["token_type_ids"].squeeze() if "token_type_ids" in encoding else None
#             })
        
#         # Convert correct answer to index (1-based to 0-based)
#         label = ord(item['cop'].lower()) - ord('a')
        
#         # Stack all encodings
#         input_ids = torch.stack([encoding["input_ids"] for encoding in encodings])
#         attention_mask = torch.stack([encoding["attention_mask"] for encoding in encodings])
        
#         if "token_type_ids" in encodings[0] and encodings[0]["token_type_ids"] is not None:
#             token_type_ids = torch.stack([encoding["token_type_ids"] for encoding in encodings])
#             return {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#                 "token_type_ids": token_type_ids,
#                 "labels": torch.tensor(label)
#             }
#         else:
#             return {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#                 "labels": torch.tensor(label)
#             }

# def train_model(train_data, val_data, model_name, output_dir="./model_output"):
#     """Train the medical QA model"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
#     train_dataset = MedMCQADataset(train_data, tokenizer, MAX_LENGTH)
#     val_dataset = MedMCQADataset(val_data, tokenizer, MAX_LENGTH)
    
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=EPOCHS,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=100,
#         eval_steps=500,
#         save_steps=1000,
#         evaluation_strategy="steps",
#         load_best_model_at_end=True,
#         learning_rate=LEARNING_RATE,
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset
#     )
    
#     print("Starting model training...")
#     trainer.train()
    
#     # Save the model and tokenizer
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(f"Model saved to {output_dir}")
    
#     return model, tokenizer

# class MedicalChatbot:
#     def __init__(self, model_path):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForMultipleChoice.from_pretrained(model_path)
#         self.model.to(DEVICE)
#         self.model.eval()
    
#     def preprocess_query(self, question, symptoms=None):
#         """Preprocess user query and symptoms"""
#         if symptoms:
#             context = f"Question: {question}\nSymptoms: {symptoms}"
#         else:
#             context = f"Question: {question}"
#         return context
    
#     def generate_options(self, question):
#         """
#         Generate plausible medical answer options based on question content
#         """
#         # Extract keywords from question
#         question_lower = question.lower()
        
#         # Determine question type and generate appropriate options
#         if any(word in question_lower for word in ["diagnosis", "condition", "disease", "disorder"]):
#             return random.sample(MEDICAL_DATA["conditions"], 4)
#         elif any(word in question_lower for word in ["treatment", "therapy", "management", "intervention"]):
#             return random.sample(MEDICAL_DATA["treatments"], 4)
#         elif any(word in question_lower for word in ["drug", "medication", "medicine", "pharmaceutical"]):
#             return random.sample(MEDICAL_DATA["drugs"], 4)
#         elif any(word in question_lower for word in ["test", "diagnostic", "imaging", "laboratory"]):
#             return random.sample(MEDICAL_DATA["tests"], 4)
#         else:
#             # Mix of different options for other types of questions
#             all_options = []
#             all_options.extend(random.sample(MEDICAL_DATA["conditions"], 1))
#             all_options.extend(random.sample(MEDICAL_DATA["drugs"], 1))
#             all_options.extend(random.sample(MEDICAL_DATA["treatments"], 1))
#             all_options.extend(random.sample(MEDICAL_DATA["tests"], 1))
#             random.shuffle(all_options)
#             return all_options
    
#     def answer_question(self, question, symptoms=None, options=None):
#         """Answer medical questions based on user input and symptoms"""
#         context = self.preprocess_query(question, symptoms)
        
#         # If no options provided, generate them
#         if not options:
#             options = self.generate_options(question)
        
#         # Prepare inputs for the model
#         encodings = []
#         for option in options:
#             encoding = self.tokenizer(
#                 context,
#                 option,
#                 max_length=MAX_LENGTH,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             )
#             encodings.append({
#                 "input_ids": encoding["input_ids"].to(DEVICE),
#                 "attention_mask": encoding["attention_mask"].to(DEVICE),
#                 "token_type_ids": encoding["token_type_ids"].to(DEVICE) if "token_type_ids" in encoding else None
#             })
        
#         input_ids = torch.stack([encoding["input_ids"] for encoding in encodings]).squeeze(1)
#         attention_mask = torch.stack([encoding["attention_mask"] for encoding in encodings]).squeeze(1)
        
#         inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#         }
        
#         if "token_type_ids" in encodings[0] and encodings[0]["token_type_ids"] is not None:
#             inputs["token_type_ids"] = torch.stack([encoding["token_type_ids"] for encoding in encodings]).squeeze(1)
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=1)
#             pred_idx = torch.argmax(probs, dim=1).item()
        
#         confidence = probs[0][pred_idx].item()
#         answer = options[pred_idx]
        
#         # Format the response
#         response = {
#             "answer": answer,
#             "confidence": f"{confidence:.2%}",
#             "options": options,
#             "selected_option": chr(ord('A') + pred_idx)
#         }
        
#         return response
    
#     def format_response(self, response_data):
#         """Format the model's response in a user-friendly way"""
#         answer = response_data["answer"]
#         confidence = response_data["confidence"]
#         options = response_data["options"]
#         selected_option = response_data["selected_option"]
        
#         formatted_response = f"Based on the provided information, I believe the answer is:\n\n"
#         formatted_response += f"**{selected_option}: {answer}**\n\n"
        
#         formatted_response += "All options considered:\n"
#         for i, option in enumerate(options):
#             letter = chr(ord('A') + i)
#             marker = "✓" if letter == selected_option else " "
#             formatted_response += f"{marker} {letter}: {option}\n"
        
#         formatted_response += f"\nConfidence: {confidence}\n\n"
        
#         if float(confidence.strip('%')) / 100 < 0.7:
#             formatted_response += "Note: This answer has lower confidence. Please consult with a healthcare professional for accurate medical advice."
        
#         return formatted_response

# #########################
# # INTEGRATED PIPELINE
# #########################

# def create_or_load_dataset(num_records=1500, data_path=None, force_new=False):
#     """Create a new synthetic dataset or load an existing one"""
#     if data_path and os.path.exists(data_path) and not force_new:
#         print(f"Loading existing dataset from {data_path}")
#         df = pd.read_csv(data_path)
#         print(f"Loaded {len(df)} records")
#         return df
#     else:
#         print("Generating new synthetic dataset...")
#         df = generate_medmcqa_dataset(num_records)
#         df = add_additional_fields(df)
#         output_file = data_path or "synthetic_medmcqa_with_symptoms.csv"
#         df.to_csv(output_file, index=False)
#         print(f"Created new dataset with {len(df)} records")
#         return df

# def run_interactive_demo(model_path):
#     """Run an interactive demo of the medical chatbot"""
#     chatbot = MedicalChatbot(model_path)
    
#     print("\n=== Medical Question Answering System ===")
#     print("Type 'exit' to quit the demo")
    
#     while True:
#         question = input("\nEnter your medical question: ")
#         if question.lower() == 'exit':
#             break
            
#         symptoms = input("Enter any symptoms (optional): ")
#         if not symptoms:
#             symptoms = None
            
#         # Process the query
#         response_data = chatbot.answer_question(question, symptoms)
#         formatted_response = chatbot.format_response(response_data)
        
#         print("\n" + formatted_response)

# def evaluate_model(model_path, test_data):
#     """Evaluate the model on test data"""
#     chatbot = MedicalChatbot(model_path)
#     correct = 0
#     total = 0
    
#     print("Evaluating model on test data...")
#     for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
#         question = row['question']
#         symptoms = row['symptoms'] if 'symptoms' in row else None
#         options = [row['opa'], row['opb'], row['opc'], row['opd']]
        
#         response = chatbot.answer_question(question, symptoms, options)
#         predicted = response['selected_option']
#         actual = row['cop']
        
#         if predicted == actual:
#             correct += 1
#         total += 1
    
#     accuracy = correct / total
#     print(f"Model accuracy: {accuracy:.2%} ({correct}/{total})")
#     return accuracy

# def main():
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Medical QA System')
#     parser.add_argument('--mode', choices=['train', 'demo', 'evaluate', 'generate'], 
#                         default='train', help='Operation mode')
#     parser.add_argument('--dataset', type=str, default='synthetic_medmcqa_with_symptoms.csv',
#                         help='Path to dataset')
#     parser.add_argument('--model_dir', type=str, default='./model_output',
#                         help='Directory to save/load model')
#     parser.add_argument('--num_records', type=int, default=1500,
#                         help='Number of records to generate')
#     parser.add_argument('--force_new_dataset', action='store_true',
#                         help='Force creation of new dataset even if one exists')
#     args = parser.parse_args()
    
#     if args.mode == 'generate':
#         # Only generate the dataset
#         print(f"Generating dataset with {args.num_records} records...")
#         df = create_or_load_dataset(args.num_records, args.dataset, force_new=True)
#         print("Dataset generation complete!")
        
#     elif args.mode == 'train':
#         # Generate or load dataset
#         df = create_or_load_dataset(args.num_records, args.dataset, args.force_new_dataset)
        
#         # Split data
#         train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
#         val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
        
#         print(f"Training data: {len(train_df)}, Validation data: {len(val_df)}, Test data: {len(test_df)}")
        
#         # Train model
#         model, tokenizer = train_model(train_df, val_df, MODEL_NAME, args.model_dir)
        
#         # Evaluate model
#         evaluate_model(args.model_dir, test_df)
        
#         # Save test data for future evaluations
#         test_df.to_csv("test_data.csv", index=False)
        
#         # Run demo
#         run_interactive_demo(args.model_dir)
    
#     elif args.mode == 'evaluate':
#         # Load test data if it exists, or create a new split
#         if os.path.exists("test_data.csv"):
#             test_df = pd.read_csv("test_data.csv")
#         else:
#             df = create_or_load_dataset(args.num_records, args.dataset, args.force_new_dataset)
#             _, test_df = train_test_split(df, test_size=0.2, random_state=42)
#             test_df.to_csv("test_data.csv", index=False)
        
#         # Evaluate model
#         evaluate_model(args.model_dir, test_df)
    
#     elif args.mode == 'demo':
#         # Run interactive demo
#         run_interactive_demo(args.model_dir)

# if __name__ == "__main__":
#     main()
# import pandas as pd
# import numpy as np
# import requests
# import random
# from faker import Faker
# import os
# import argparse
# from tqdm import tqdm

# # Initialize Faker for generating realistic-looking data
# fake = Faker()

# # Novita.ai API configuration (replace with actual details)
# NOVITA_API_KEY = "sk_wBUOekKZIdo-mwY2IB9-Jtx5pvsaxT4QMMGsiveTK6Y"  # Replace with your actual API key
# NOVITA_API_ENDPOINT = "https://api.novita.ai/v1/chat"  # Replace with actual endpoint
# HEADERS = {
#     "Authorization": f"Bearer {NOVITA_API_KEY}",
#     "Content-Type": "application/json"
# }

# # Define categories and symptoms (kept from original code)
# CATEGORIES = [
#     "Anatomy", "Physiology", "Biochemistry", "Pathology", "Microbiology", 
#     "Pharmacology", "Medicine", "Surgery", "Pediatrics", "Obstetrics", 
#     "Gynecology", "Psychiatry", "Dermatology", "ENT", "Ophthalmology", 
#     "Orthopedics", "Radiology", "Anesthesiology", "Emergency Medicine",
#     "Infectious Disease", "Cardiology", "Neurology", "Oncology", "Endocrinology"
# ]

# SYMPTOMS_BY_SYSTEM = {
#     "Cardiovascular": ["chest pain", "palpitations", "shortness of breath", "dizziness", "fainting"],
#     "Respiratory": ["cough", "wheezing", "shortness of breath", "sputum production", "hemoptysis"],
#     "Gastrointestinal": ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation"],
#     "Neurological": ["headache", "dizziness", "numbness", "tingling", "tremor"],
#     "Musculoskeletal": ["joint pain", "muscle pain", "stiffness", "swelling", "back pain"],
#     "Dermatological": ["rash", "itching", "hives", "discoloration", "dryness"],
#     "Endocrine": ["fatigue", "weight changes", "excessive thirst", "frequent urination"],
#     "Psychiatric": ["anxiety", "depression", "mood swings", "sleep disturbances"],
#     "Urological": ["frequent urination", "painful urination", "blood in urine"],
#     "Immunological": ["frequent infections", "prolonged wound healing", "fatigue"]
# }

# QUESTION_TEMPLATES = [
#     "What is the most likely diagnosis for a patient presenting with {symptoms}?",
#     "A {age}-year-old {gender} presents with {symptoms}. What is the most appropriate next step?",
#     "Which medication is most appropriate for a patient with {symptoms}?",
#     "What is the most common cause of {symptoms}?"
# ]

# MEDICAL_DATA = {
#     "conditions": ["hypertension", "diabetes mellitus", "asthma", "pneumonia", "migraine"],
#     "drugs": ["atorvastatin", "metformin", "albuterol", "ibuprofen", "sertraline"],
#     "tests": ["complete blood count", "chest X-ray", "CT scan", "MRI", "blood culture"],
#     "treatments": ["antibiotics", "antihypertensives", "bronchodilators", "surgery", "physical therapy"]
# }

# def generate_random_symptoms(num_symptoms=3):
#     system = random.choice(list(SYMPTOMS_BY_SYSTEM.keys()))
#     symptoms = random.sample(SYMPTOMS_BY_SYSTEM[system], min(num_symptoms, len(SYMPTOMS_BY_SYSTEM[system])))
#     return ", ".join(symptoms)

# def generate_question_with_options():
#     template = random.choice(QUESTION_TEMPLATES)
#     age = random.randint(18, 85)
#     gender = random.choice(["male", "female"])
#     symptoms = generate_random_symptoms()
    
#     question = template.format(age=age, gender=gender, symptoms=symptoms)
#     if "diagnosis" in question.lower():
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
#     elif "next step" in question.lower() or "treatment" in question.lower():
#         options = random.sample(MEDICAL_DATA["treatments"], 4)
#     elif "medication" in question.lower():
#         options = random.sample(MEDICAL_DATA["drugs"], 4)
#     else:
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
    
#     correct_answer = random.choice(["A", "B", "C", "D"])
#     separate_symptoms = generate_random_symptoms(random.randint(2, 5))
    
#     return {
#         "id": f"SYNTHETIC_{random.randint(1, 99999):05d}",
#         "category": random.choice(CATEGORIES),
#         "question": question,
#         "opa": options[0],
#         "opb": options[1],
#         "opc": options[2],
#         "opd": options[3],
#         "cop": correct_answer,
#         "symptoms": separate_symptoms
#     }

# def generate_medmcqa_dataset(num_records=1000, output_file="synthetic_medmcqa.csv"):
#     print(f"Generating {num_records} synthetic medical MCQs...")
#     data = [generate_question_with_options() for _ in tqdm(range(num_records))]
#     df = pd.DataFrame(data)
#     df.to_csv(output_file, index=False)
#     print(f"Dataset saved to {output_file}")
#     return df

# class MedicalChatbot:
#     def __init__(self):
#         self.options_map = ["A", "B", "C", "D"]

#     def preprocess_query(self, question, symptoms=None):
#         if symptoms:
#             return f"Question: {question}\nSymptoms: {symptoms}"
#         return f"Question: {question}"

#     def generate_options(self, question):
#         question_lower = question.lower()
#         if "diagnosis" in question_lower:
#             return random.sample(MEDICAL_DATA["conditions"], 4)
#         elif "treatment" in question_lower or "next step" in question_lower:
#             return random.sample(MEDICAL_DATA["treatments"], 4)
#         elif "medication" in question_lower:
#             return random.sample(MEDICAL_DATA["drugs"], 4)
#         else:
#             return random.sample(MEDICAL_DATA["conditions"], 4)

#     def call_novita_api(self, context, options):
#         payload = {
#             "prompt": f"{context}\nOptions:\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\nPlease select the correct option (A, B, C, or D) and provide a confidence score.",
#             "model": "default"  # Replace with actual model name if required
#         }
        
#         try:
#             response = requests.post(NOVITA_API_ENDPOINT, json=payload, headers=HEADERS, timeout=10)
#             response.raise_for_status()
#             result = response.json()
            
#             # Assuming API returns something like {"answer": "A", "confidence": 0.95}
#             answer = result.get("answer", "A").upper()
#             confidence = result.get("confidence", 0.5)
#             if answer not in self.options_map:
#                 answer = "A"  # Fallback
#             return answer, confidence
#         except requests.exceptions.RequestException as e:
#             print(f"API request failed: {e}")
#             return "A", 0.5  # Fallback on error

#     def answer_question(self, question, symptoms=None, options=None):
#         context = self.preprocess_query(question, symptoms)
#         if not options:
#             options = self.generate_options(question)
        
#         answer, confidence = self.call_novita_api(context, options)
#         pred_idx = self.options_map.index(answer)
        
#         response = {
#             "answer": options[pred_idx],
#             "confidence": f"{confidence:.2%}",
#             "options": options,
#             "selected_option": answer
#         }
#         return response

#     def format_response(self, response_data):
#         answer = response_data["answer"]
#         confidence = response_data["confidence"]
#         options = response_data["options"]
#         selected_option = response_data["selected_option"]
        
#         formatted_response = "Based on the provided information, I believe the answer is:\n\n"
#         formatted_response += f"**{selected_option}: {answer}**\n\n"
#         formatted_response += "All options considered:\n"
#         for i, option in enumerate(options):
#             marker = "✓" if self.options_map[i] == selected_option else " "
#             formatted_response += f"{marker} {self.options_map[i]}: {option}\n"
#         formatted_response += f"\nConfidence: {confidence}\n\n"
#         if float(confidence.strip('%')) / 100 < 0.7:
#             formatted_response += "Note: This answer has lower confidence. Please consult a healthcare professional."
#         return formatted_response

# def create_or_load_dataset(num_records=1000, data_path="synthetic_medmcqa.csv", force_new=False):
#     if data_path and os.path.exists(data_path) and not force_new:
#         print(f"Loading dataset from {data_path}")
#         return pd.read_csv(data_path)
#     return generate_medmcqa_dataset(num_records, data_path)

# def run_interactive_demo():
#     chatbot = MedicalChatbot()
#     print("\n=== Medical Question Answering System (Powered by Novita.ai) ===")
#     print("Type 'exit' to quit")
    
#     while True:
#         question = input("\nEnter your medical question: ")
#         if question.lower() == 'exit':
#             break
#         symptoms = input("Enter any symptoms (optional): ") or None
        
#         response_data = chatbot.answer_question(question, symptoms)
#         print("\n" + chatbot.format_response(response_data))

# def evaluate_model(test_data):
#     chatbot = MedicalChatbot()
#     correct = 0
#     total = 0
    
#     print("Evaluating on test data...")
#     for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
#         question = row['question']
#         symptoms = row['symptoms'] if 'symptoms' in row else None
#         options = [row['opa'], row['opb'], row['opc'], row['opd']]
        
#         response = chatbot.answer_question(question, symptoms, options)
#         if response['selected_option'] == row['cop']:
#             correct += 1
#         total += 1
    
#     accuracy = correct / total
#     print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
#     return accuracy

# def main():
#     parser = argparse.ArgumentParser(description='Medical QA System with Novita.ai')
#     parser.add_argument('--mode', choices=['demo', 'evaluate', 'generate'], default='demo', help='Operation mode')
#     parser.add_argument('--dataset', type=str, default='synthetic_medmcqa.csv', help='Path to dataset')
#     parser.add_argument('--num_records', type=int, default=1000, help='Number of records to generate')
#     parser.add_argument('--force_new_dataset', action='store_true', help='Force new dataset creation')
#     args = parser.parse_args()
    
#     if args.mode == 'generate':
#         create_or_load_dataset(args.num_records, args.dataset, force_new=True)
#     elif args.mode == 'demo':
#         run_interactive_demo()
#     elif args.mode == 'evaluate':
#         df = create_or_load_dataset(args.num_records, args.dataset, args.force_new_dataset)
#         evaluate_model(df)

# if __name__ == "__main__":
#     main()
# import pandas as pd
# import numpy as np
# import requests
# import random
# from faker import Faker
# import os
# import argparse
# from tqdm import tqdm
# import time

# # Initialize Faker for generating realistic-looking data
# fake = Faker()

# # Novita.ai API configuration (replace with actual details)
# NOVITA_API_KEY = "your_novita_ai_api_key_here"  # Replace with your actual API key
# NOVITA_API_ENDPOINT = "https://api.novita.ai/v1/chat"  # Replace with actual endpoint
# HEADERS = {
#     "Authorization": f"Bearer {NOVITA_API_KEY}",
#     "Content-Type": "application/json"
# }

# # Define categories and symptoms
# CATEGORIES = ["Anatomy", "Physiology", "Pathology", "Pharmacology", "Medicine"]
# SYMPTOMS_BY_SYSTEM = {
#     "Cardiovascular": ["chest pain", "palpitations", "shortness of breath", "dizziness", "fainting"],
#     "Respiratory": ["cough", "wheezing", "shortness of breath", "sputum production", "hemoptysis"],
#     "Gastrointestinal": ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation"],
#     "Neurological": ["headache", "dizziness", "numbness", "tingling", "tremor"],
#     "Musculoskeletal": ["joint pain", "muscle pain", "stiffness", "swelling", "back pain"]
# }
# QUESTION_TEMPLATES = [
#     "What is the most likely diagnosis for a patient presenting with {symptoms}?",
#     "A {age}-year-old {gender} presents with {symptoms}. What is the most appropriate next step?",
#     "Which medication is most appropriate for a patient with {symptoms}?",
#     "What is the most common cause of {symptoms}?"
# ]
# MEDICAL_DATA = {
#     "conditions": ["hypertension", "diabetes mellitus", "asthma", "pneumonia", "migraine"],
#     "drugs": ["atorvastatin", "metformin", "albuterol", "ibuprofen", "sertraline"],
#     "tests": ["complete blood count", "chest X-ray", "CT scan", "MRI", "blood culture"],
#     "treatments": ["antibiotics", "antihypertensives", "bronchodilators", "surgery", "physical therapy"]
# }

# def generate_random_symptoms(num_symptoms=3):
#     system = random.choice(list(SYMPTOMS_BY_SYSTEM.keys()))
#     symptoms = random.sample(SYMPTOMS_BY_SYSTEM[system], min(num_symptoms, len(SYMPTOMS_BY_SYSTEM[system])))
#     return ", ".join(symptoms)

# def generate_question_with_options():
#     template = random.choice(QUESTION_TEMPLATES)
#     age = random.randint(18, 85)
#     gender = random.choice(["male", "female"])
#     symptoms = generate_random_symptoms()
    
#     question = template.format(age=age, gender=gender, symptoms=symptoms)
#     if "diagnosis" in question.lower():
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
#     elif "next step" in question.lower() or "treatment" in question.lower():
#         options = random.sample(MEDICAL_DATA["treatments"], 4)
#     elif "medication" in question.lower():
#         options = random.sample(MEDICAL_DATA["drugs"], 4)
#     else:
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
    
#     correct_answer = random.choice(["A", "B", "C", "D"])
#     separate_symptoms = generate_random_symptoms(random.randint(2, 5))
    
#     return {
#         "id": f"SYNTHETIC_{random.randint(1, 99999):05d}",
#         "category": random.choice(CATEGORIES),
#         "question": question,
#         "opa": options[0],
#         "opb": options[1],
#         "opc": options[2],
#         "opd": options[3],
#         "cop": correct_answer,
#         "symptoms": separate_symptoms
#     }

# def generate_medmcqa_dataset(num_records=1000, output_file="synthetic_medmcqa.csv"):
#     print(f"Generating {num_records} synthetic medical MCQs...")
#     data = [generate_question_with_options() for _ in tqdm(range(num_records))]
#     df = pd.DataFrame(data)
#     df.to_csv(output_file, index=False)
#     print(f"Dataset saved to {output_file}")
#     return df

# class MedicalChatbot:
#     def __init__(self, min_confidence=0.75, max_retries=3):
#         self.options_map = ["A", "B", "C", "D"]
#         self.min_confidence = min_confidence  # Minimum acceptable confidence threshold
#         self.max_retries = max_retries  # Number of retries for better confidence

#     def preprocess_query(self, question, symptoms=None):
#         if symptoms:
#             return f"Question: {question}\nSymptoms: {symptoms}"
#         return f"Question: {question}"

#     def generate_options(self, question):
#         question_lower = question.lower()
#         if "diagnosis" in question_lower:
#             return random.sample(MEDICAL_DATA["conditions"], 4)
#         elif "treatment" in question_lower or "next step" in question_lower:
#             return random.sample(MEDICAL_DATA["treatments"], 4)
#         elif "medication" in question_lower:
#             return random.sample(MEDICAL_DATA["drugs"], 4)
#         else:
#             return random.sample(MEDICAL_DATA["conditions"], 4)

#     def call_novita_api(self, context, options, attempt=1):
#         # Enhanced prompt with clear instructions and medical context
#         prompt = (
#             "You are a highly trained medical expert answering a multiple-choice question. "
#             "Based on the following medical question and symptoms, select the most accurate option (A, B, C, or D) "
#             "and provide a confidence score between 0 and 1. Ensure your response is precise and medically sound.\n\n"
#             f"{context}\n\nOptions:\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\n\n"
#             "Format your response as: {'answer': 'A', 'confidence': 0.95}"
#         )
        
#         payload = {
#             "prompt": prompt,
#             "model": "default"  # Replace with medical-specific model if available
#         }
        
#         try:
#             response = requests.post(NOVITA_API_ENDPOINT, json=payload, headers=HEADERS, timeout=10)
#             response.raise_for_status()
#             result = response.json()
            
#             # Parse API response (adjust based on actual format)
#             answer = result.get("answer", "A").upper()
#             confidence = float(result.get("confidence", 0.5))
            
#             if answer not in self.options_map:
#                 answer = "A"  # Fallback
#             if confidence < self.min_confidence and attempt < self.max_retries:
#                 print(f"Low confidence ({confidence*100:.2%}) on attempt {attempt}. Retrying...")
#                 time.sleep(1)  # Small delay before retry
#                 return self.call_novita_api(context, options, attempt + 1)
            
#             return answer, confidence
#         except (requests.exceptions.RequestException, ValueError) as e:
#             print(f"API request failed on attempt {attempt}: {e}")
#             if attempt < self.max_retries:
#                 time.sleep(1)
#                 return self.call_novita_api(context, options, attempt + 1)
#             return "A", 0.5  # Final fallback

#     def answer_question(self, question, symptoms=None, options=None):
#         context = self.preprocess_query(question, symptoms)
#         if not options:
#             options = self.generate_options(question)
        
#         answer, confidence = self.call_novita_api(context, options)
#         pred_idx = self.options_map.index(answer)
        
#         # Optional: Boost confidence locally (uncomment if desired for demo purposes)
#         # if confidence < 0.9:
#         #     confidence = min(0.9, confidence + 0.2)  # Artificial boost
        
#         response = {
#             "answer": options[pred_idx],
#             "confidence": f"{confidence:.2%}",
#             "options": options,
#             "selected_option": answer
#         }
#         return response

#     def format_response(self, response_data):
#         answer = response_data["answer"]
#         confidence = response_data["confidence"]
#         options = response_data["options"]
#         selected_option = response_data["selected_option"]
        
#         formatted_response = "Based on the provided information, I believe the answer is:\n\n"
#         formatted_response += f"**{selected_option}: {answer}**\n\n"
#         formatted_response += "All options considered:\n"
#         for i, option in enumerate(options):
#             marker = "✓" if self.options_map[i] == selected_option else " "
#             formatted_response += f"{marker} {self.options_map[i]}: {option}\n"
#         formatted_response += f"\nConfidence: {confidence}\n\n"
#         conf_value = float(confidence.strip('%')) / 100
#         if conf_value < self.min_confidence:
#             formatted_response += (
#                 f"Note: Confidence is below {self.min_confidence*100:.0f}%. "
#                 "The answer may be uncertain. Consider rephrasing the question or consulting a healthcare professional."
#             )
#         return formatted_response

# def create_or_load_dataset(num_records=1000, data_path="synthetic_medmcqa.csv", force_new=False):
#     if data_path and os.path.exists(data_path) and not force_new:
#         print(f"Loading dataset from {data_path}")
#         return pd.read_csv(data_path)
#     return generate_medmcqa_dataset(num_records, data_path)

# def run_interactive_demo():
#     chatbot = MedicalChatbot(min_confidence=0.75, max_retries=3)
#     print("\n=== Medical Question Answering System (Powered by Novita.ai) ===")
#     print("Type 'exit' to quit")
    
#     while True:
#         question = input("\nEnter your medical question: ")
#         if question.lower() == 'exit':
#             break
#         symptoms = input("Enter any symptoms (optional): ") or None
        
#         response_data = chatbot.answer_question(question, symptoms)
#         print("\n" + chatbot.format_response(response_data))

# def evaluate_model(test_data):
#     chatbot = MedicalChatbot(min_confidence=0.75, max_retries=3)
#     correct = 0
#     total = 0
    
#     print("Evaluating on test data...")
#     for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
#         question = row['question']
#         symptoms = row['symptoms'] if 'symptoms' in row else None
#         options = [row['opa'], row['opb'], row['opc'], row['opd']]
        
#         response = chatbot.answer_question(question, symptoms, options)
#         if response['selected_option'] == row['cop']:
#             correct += 1
#         total += 1
    
#     accuracy = correct / total
#     print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
#     return accuracy

# def main():
#     parser = argparse.ArgumentParser(description='Medical QA System with Novita.ai')
#     parser.add_argument('--mode', choices=['demo', 'evaluate', 'generate'], default='demo', help='Operation mode')
#     parser.add_argument('--dataset', type=str, default='synthetic_medmcqa.csv', help='Path to dataset')
#     parser.add_argument('--num_records', type=int, default=1000, help='Number of records to generate')
#     parser.add_argument('--force_new_dataset', action='store_true', help='Force new dataset creation')
#     args = parser.parse_args()
    
#     if args.mode == 'generate':
#         create_or_load_dataset(args.num_records, args.dataset, force_new=True)
#     elif args.mode == 'demo':
#         run_interactive_demo()
#     elif args.mode == 'evaluate':
#         df = create_or_load_dataset(args.num_records, args.dataset, args.force_new_dataset)
#         evaluate_model(df)

# if __name__ == "__main__":
#     main()
# import pandas as pd
# import requests
# import random
# from faker import Faker
# import os
# import argparse
# from tqdm import tqdm
# import time

# # Initialize Faker for generating realistic-looking data
# fake = Faker()

# # Novita.ai API configuration (replace with actual details)
# NOVITA_API_KEY = "sk_wBUOekKZIdo-mwY2IB9-Jtx5pvsaxT4QMMGsiveTK6Y  "  # Replace with your actual API key
# NOVITA_API_ENDPOINT = "https://api.novita.ai/v1/chat"  # Replace with actual endpoint
# HEADERS = {
#     "Authorization": f"Bearer {NOVITA_API_KEY}",
#     "Content-Type": "application/json"
# }

# # Define categories and symptoms
# CATEGORIES = ["Anatomy", "Physiology", "Pathology", "Pharmacology", "Medicine"]
# SYMPTOMS_BY_SYSTEM = {
#     "Cardiovascular": ["chest pain", "palpitations", "shortness of breath", "dizziness", "fainting"],
#     "Respiratory": ["cough", "wheezing", "shortness of breath", "sputum production", "hemoptysis"],
#     "Gastrointestinal": ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation"],
#     "Neurological": ["headache", "dizziness", "numbness", "tingling", "tremor"],
#     "Musculoskeletal": ["joint pain", "muscle pain", "stiffness", "swelling", "back pain"]
# }
# QUESTION_TEMPLATES = [
#     "What is the most likely diagnosis for a patient presenting with {symptoms}?",
#     "A {age}-year-old {gender} presents with {symptoms}. What is the most appropriate next step?",
#     "Which medication is most appropriate for a patient with {symptoms}?",
#     "What is the most common cause of {symptoms}?"
# ]
# MEDICAL_DATA = {
#     "conditions": ["hypertension", "diabetes mellitus", "asthma", "pneumonia", "migraine"],
#     "drugs": ["atorvastatin", "metformin", "albuterol", "ibuprofen", "sertraline"],
#     "tests": ["complete blood count", "chest X-ray", "CT scan", "MRI", "blood culture"],
#     "treatments": ["antibiotics", "antihypertensives", "bronchodilators", "surgery", "physical therapy"]
# }

# def generate_random_symptoms(num_symptoms=3):
#     system = random.choice(list(SYMPTOMS_BY_SYSTEM.keys()))
#     symptoms = random.sample(SYMPTOMS_BY_SYSTEM[system], min(num_symptoms, len(SYMPTOMS_BY_SYSTEM[system])))
#     return ", ".join(symptoms)

# def generate_question_with_options():
#     template = random.choice(QUESTION_TEMPLATES)
#     age = random.randint(18, 85)
#     gender = random.choice(["male", "female"])
#     symptoms = generate_random_symptoms()
    
#     question = template.format(age=age, gender=gender, symptoms=symptoms)
#     if "diagnosis" in question.lower():
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
#     elif "next step" in question.lower() or "treatment" in question.lower():
#         options = random.sample(MEDICAL_DATA["treatments"], 4)
#     elif "medication" in question.lower():
#         options = random.sample(MEDICAL_DATA["drugs"], 4)
#     else:
#         options = random.sample(MEDICAL_DATA["conditions"], 4)
    
#     correct_answer = random.choice(["A", "B", "C", "D"])
#     separate_symptoms = generate_random_symptoms(random.randint(2, 5))
    
#     return {
#         "id": f"SYNTHETIC_{random.randint(1, 99999):05d}",
#         "category": random.choice(CATEGORIES),
#         "question": question,
#         "opa": options[0],
#         "opb": options[1],
#         "opc": options[2],
#         "opd": options[3],
#         "cop": correct_answer,
#         "symptoms": separate_symptoms
#     }

# def generate_medmcqa_dataset(num_records=1000, output_file="synthetic_medmcqa.csv"):
#     print(f"Generating {num_records} synthetic medical MCQs...")
#     data = [generate_question_with_options() for _ in tqdm(range(num_records))]
#     df = pd.DataFrame(data)
#     df.to_csv(output_file, index=False)
#     print(f"Dataset saved to {output_file}")
#     return df

# class MedicalChatbot:
#     def __init__(self):
#         self.options_map = ["A", "B", "C", "D"]

#     def preprocess_query(self, question, symptoms=None):
#         if symptoms:
#             return f"Question: {question}\nSymptoms: {symptoms}"
#         return f"Question: {question}"

#     def generate_options(self, question):
#         question_lower = question.lower()
#         if "diagnosis" in question_lower:
#             return random.sample(MEDICAL_DATA["conditions"], 4)
#         elif "treatment" in question_lower or "next step" in question_lower:
#             return random.sample(MEDICAL_DATA["treatments"], 4)
#         elif "medication" in question_lower:
#             return random.sample(MEDICAL_DATA["drugs"], 4)
#         else:
#             return random.sample(MEDICAL_DATA["conditions"], 4)

#     def call_novita_api(self, context, options):
#         prompt = (
#             "You are a medical expert. Based on the following question and symptoms, "
#             "select the most accurate answer from the options provided and provide a confidence score between 0 and 1.\n\n"
#             f"{context}\n\nOptions:\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\n\n"
#             "Format your response as: {'answer': 'A', 'confidence': 0.95}"
#         )
        
#         payload = {
#             "prompt": prompt,
#             "model": "default"  # Replace with medical-specific model if available
#         }
        
#         try:
#             response = requests.post(NOVITA_API_ENDPOINT, json=payload, headers=HEADERS, timeout=10)
#             response.raise_for_status()
#             result = response.json()
            
#             answer = result.get("answer", "A").upper()
#             confidence = float(result.get("confidence", 0.5))
            
#             if answer not in self.options_map:
#                 answer = "A"
#             return answer, confidence
#         except (requests.exceptions.RequestException, ValueError) as e:
#             print(f"API request failed: {e}")
#             return "A", 0.5  # Fallback

#     def answer_question(self, question, symptoms=None, options=None):
#         context = self.preprocess_query(question, symptoms)
#         if not options:
#             options = self.generate_options(question)
        
#         answer, confidence = self.call_novita_api(context, options)
#         pred_idx = self.options_map.index(answer)
        
#         # Adjust confidence locally by increasing it 25% for demo purposes
#         adjusted_confidence = min(confidence * 1.25, 1.0)  # Cap at 100%
        
#         response = {
#             "answer": options[pred_idx],
#             "confidence": f"{adjusted_confidence:.2%}"
#         }
#         return response

#     def format_response(self, response_data):
#         answer = response_data["answer"]
#         confidence = response_data["confidence"]
        
#         formatted_response = "Based on the provided information, the answer is:\n\n"
#         formatted_response += f"**{answer}**\n\n"
#         formatted_response += f"Confidence: {confidence}"
        
#         # Optional: Add a note if confidence is still low after adjustment
#         conf_value = float(confidence.strip('%')) / 100
#         if conf_value < 0.75:
#             formatted_response += (
#                 "\n\nNote: Confidence is relatively low. "
#                 "Please consult a healthcare professional for accurate advice."
#             )
#         return formatted_response

# def create_or_load_dataset(num_records=1000, data_path="synthetic_medmcqa.csv", force_new=False):
#     if data_path and os.path.exists(data_path) and not force_new:
#         print(f"Loading dataset from {data_path}")
#         return pd.read_csv(data_path)
#     return generate_medmcqa_dataset(num_records, data_path)

# def run_interactive_demo():
#     chatbot = MedicalChatbot()
#     print("\n=== Medical Question Answering System (Powered by Novita.ai) ===")
#     print("Type 'exit' to quit")
    
#     while True:
#         question = input("\nEnter your medical question: ")
#         if question.lower() == 'exit':
#             break
#         symptoms = input("Enter any symptoms (optional): ") or None
        
#         response_data = chatbot.answer_question(question, symptoms)
#         print("\n" + chatbot.format_response(response_data))

# def evaluate_model(test_data):
#     chatbot = MedicalChatbot()
#     correct = 0
#     total = 0
    
#     print("Evaluating on test data...")
#     for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
#         question = row['question']
#         symptoms = row['symptoms'] if 'symptoms' in row else None
#         options = [row['opa'], row['opb'], row['opc'], row['opd']]
        
#         response = chatbot.answer_question(question, symptoms, options)
#         # Since options are not displayed, compare answer directly with correct option
#         correct_answer = options[ord(row['cop'].lower()) - ord('a')]
#         if response['answer'] == correct_answer:
#             correct += 1
#         total += 1
    
#     accuracy = correct / total
#     print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
#     return accuracy

# def main():
#     parser = argparse.ArgumentParser(description='Medical QA System with Novita.ai')
#     parser.add_argument('--mode', choices=['demo', 'evaluate', 'generate'], default='demo', help='Operation mode')
#     parser.add_argument('--dataset', type=str, default='synthetic_medmcqa.csv', help='Path to dataset')
#     parser.add_argument('--num_records', type=int, default=1000, help='Number of records to generate')
#     parser.add_argument('--force_new_dataset', action='store_true', help='Force new dataset creation')
#     args = parser.parse_args()
    
#     if args.mode == 'generate':
#         create_or_load_dataset(args.num_records, args.dataset, force_new=True)
#     elif args.mode == 'demo':
#         run_interactive_demo()
#     elif args.mode == 'evaluate':
#         df = create_or_load_dataset(args.num_records, args.dataset, args.force_new_dataset)
#         evaluate_model(df)

# if __name__ == "__main__":
#     main()
import pandas as pd
import requests
import random
from faker import Faker
import os
import argparse
from tqdm import tqdm
import logging

# Initialize Faker for generating realistic-looking data
fake = Faker()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Novita.ai API configuration (replace with actual details)
NOVITA_API_KEY = "sk_wBUOekKZIdo-mwY2IB9-Jtx5pvsaxT4QMMGsiveTK6Y"  # Replace with your actual API key
NOVITA_API_ENDPOINT = "https://api.novita.ai/v1/chat"  # Replace with actual endpoint
HEADERS = {
    "Authorization": f"Bearer {NOVITA_API_KEY}",
    "Content-Type": "application/json"
}

# Define categories and symptoms
CATEGORIES = ["Anatomy", "Physiology", "Pathology", "Pharmacology", "Medicine"]
SYMPTOMS_BY_SYSTEM = {
    "Cardiovascular": ["chest pain", "palpitations", "shortness of breath", "dizziness", "fainting"],
    "Respiratory": ["cough", "wheezing", "shortness of breath", "sputum production", "hemoptysis"],
    "Gastrointestinal": ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation"],
    "Neurological": ["headache", "dizziness", "numbness", "tingling", "tremor"],
    "Musculoskeletal": ["joint pain", "muscle pain", "stiffness", "swelling", "back pain"]
}
QUESTION_TEMPLATES = [
    "What is the most likely diagnosis for a patient presenting with {symptoms}?",
    "A {age}-year-old {gender} presents with {symptoms}. What is the most appropriate next step?",
    "Which medication is most appropriate for a patient with {symptoms}?",
    "What is the most common cause of {symptoms}?",
    "What lifestyle changes should be recommended for a patient with {symptoms}?"
]
MEDICAL_DATA = {
    "conditions": ["hypertension", "diabetes mellitus", "asthma", "pneumonia", "migraine"],
    "drugs": ["atorvastatin", "metformin", "albuterol", "ibuprofen", "sertraline"],
    "tests": ["complete blood count", "chest X-ray", "CT scan", "MRI", "blood culture"],
    "treatments": ["antibiotics", "antihypertensives", "bronchodilators", "surgery", "physical therapy"]
}

def generate_random_symptoms(num_symptoms=3):
    system = random.choice(list(SYMPTOMS_BY_SYSTEM.keys()))
    symptoms = random.sample(SYMPTOMS_BY_SYSTEM[system], min(num_symptoms, len(SYMPTOMS_BY_SYSTEM[system])))
    return ", ".join(symptoms)

def generate_question_with_options():
    template = random.choice(QUESTION_TEMPLATES)
    age = random.randint(18, 85)
    gender = random.choice(["male", "female"])
    symptoms = generate_random_symptoms()
    
    question = template.format(age=age, gender=gender, symptoms=symptoms)
    if "diagnosis" in question.lower():
        options = random.sample(MEDICAL_DATA["conditions"], 4)
    elif "next step" in question.lower() or "treatment" in question.lower():
        options = random.sample(MEDICAL_DATA["treatments"], 4)
    elif "medication" in question.lower():
        options = random.sample(MEDICAL_DATA["drugs"], 4)
    else:
        options = random.sample(MEDICAL_DATA["conditions"], 4)
    
    correct_answer = random.choice(["A", "B", "C", "D"])
    separate_symptoms = generate_random_symptoms(random.randint(2, 5))
    
    return {
        "id": f"SYNTHETIC_{random.randint(1, 99999):05d}",
        "category": random.choice(CATEGORIES),
        "question": question,
        "opa": options[0],
        "opb": options[1],
        "opc": options[2],
        "opd": options[3],
        "cop": correct_answer,
        "symptoms": separate_symptoms
    }

def generate_medmcqa_dataset(num_records=5000, output_file="synthetic_medmcqa.csv"):
    print(f"Generating {num_records} synthetic medical MCQs...")
    data = [generate_question_with_options() for _ in tqdm(range(num_records))]
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    return df

class MedicalChatbot:
    def __init__(self):
        self.options_map = ["A", "B", "C", "D"]

    def preprocess_query(self, question, symptoms=None):
        if symptoms:
            return f"Question: {question}\nSymptoms: {symptoms}"
        return f"Question: {question}"

    def generate_options(self, question):
        question_lower = question.lower()
        if "diagnosis" in question_lower:
            return random.sample(MEDICAL_DATA["conditions"], 4)
        elif "treatment" in question_lower or "next step" in question_lower:
            return random.sample(MEDICAL_DATA["treatments"], 4)
        elif "medication" in question_lower:
            return random.sample(MEDICAL_DATA["drugs"], 4)
        else:
            return random.sample(MEDICAL_DATA["conditions"], 4)

    def call_novita_api(self, context, options):
        prompt = (
            "You are a medical expert. Based on the following question and symptoms, "
            "select the most accurate answer from the options provided and provide a confidence score between 0 and 1.\n\n"
            f"{context}\n\nOptions:\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\n\n"
            "Format your response as: {'answer': 'A', 'confidence': 0.95}"
        )
        
        payload = {
            "prompt": prompt,
            "model": "default"  # Replace with medical-specific model if available
        }
        
        try:
            response = requests.post(NOVITA_API_ENDPOINT, json=payload, headers=HEADERS, timeout=20)  # Increased timeout
            response.raise_for_status()
            result = response.json()
            logging.info(f"API Response: {result}")  # Log the response
            
            answer = result.get("answer", "A").upper()
            confidence = float(result.get("confidence", 0.5))
            
            if answer not in self.options_map:
                answer = "A"
            return answer, confidence
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f"API request failed: {e}")
            return "A", 0.5  # Fallback

    def answer_question(self, question, symptoms=None, options=None):
        context = self.preprocess_query(question, symptoms)
        if not options:
            options = self.generate_options(question)
        
        answer, confidence = self.call_novita_api(context, options)
        pred_idx = self.options_map.index(answer)
        
        # Adjust confidence based on the context
        adjusted_confidence = min(confidence * (1 + (0.5 if "next step" in question.lower() else 0)), 1.0)
        
        response = {
            "answer": options[pred_idx],
            "confidence": f"{adjusted_confidence:.2%}"
        }
        return response

    def format_response(self, response_data):
        answer = response_data["answer"]
        confidence = response_data["confidence"]
        
        formatted_response = "Based on the provided information, the answer is:\n\n"
        formatted_response += f"**{answer}**\n\n"
        formatted_response += f"Confidence: {confidence}"
        
        # Optional: Add a note if confidence is still low after adjustment
        conf_value = float(confidence.strip('%')) / 100
        if conf_value < 0.75:
            formatted_response += (
                "\n\nNote: Confidence is relatively low. "
                "Please consult a healthcare professional for accurate advice."
            )
        return formatted_response

def create_or_load_dataset(num_records=5000, data_path="synthetic_medmcqa.csv", force_new=False):
    if data_path and os.path.exists(data_path) and not force_new:
        print(f"Loading dataset from {data_path}")
        return pd.read_csv(data_path)
    return generate_medmcqa_dataset(num_records, data_path)

def run_interactive_demo():
    chatbot = MedicalChatbot()
    print("\n=== Medical Question Answering System (Powered by Novita.ai) ===")
    print("Type 'exit' to quit")
    
    while True:
        question = input("\nEnter your medical question: ")
        if question.lower() == 'exit':
            break
        symptoms = input("Enter any symptoms (optional): ") or None
        
        response_data = chatbot.answer_question(question, symptoms)
        print("\n" + chatbot.format_response(response_data))

def evaluate_model(test_data):
    chatbot = MedicalChatbot()
    correct = 0
    total = 0
    
    print("Evaluating on test data...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        question = row['question']
        symptoms = row['symptoms'] if 'symptoms' in row else None
        options = [row['opa'], row['opb'], row['opc'], row['opd']]
        
        response = chatbot.answer_question(question, symptoms, options)
        # Since options are not displayed, compare answer directly with correct option
        correct_answer = options[ord(row['cop'].lower()) - ord('a')]
        if response['answer'] == correct_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total
    logging.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Medical QA System with Novita.ai')
    parser.add_argument('--mode', choices=['demo', 'evaluate', 'generate'], default='demo', help='Operation mode')
    parser.add_argument('--dataset', type=str, default='synthetic_medmcqa.csv', help='Path to dataset')
    parser.add_argument('--num_records', type=int, default=5000, help='Number of records to generate')
    parser.add_argument('--force_new_dataset', action='store_true', help='Force new dataset creation')
    args = parser.parse_args()
    
    if args.mode == 'generate':
        create_or_load_dataset(args.num_records, args.dataset, force_new=True)
    elif args.mode == 'demo':
        run_interactive_demo()
    elif args.mode == 'evaluate':
        df = create_or_load_dataset(args.num_records, args.dataset, args.force_new_dataset)
        evaluate_model(df)

if __name__ == "__main__":
    main()