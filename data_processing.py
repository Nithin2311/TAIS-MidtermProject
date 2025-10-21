# data_processor.py - Data loading and preprocessing
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os

class ResumePreprocessor:
    """Text preprocessing pipeline for resumes"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    
    def clean_text(self, text):
        """Clean and normalize resume text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\']', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Length normalization (first 600 + last 200 words)
        words = text.split()
        if len(words) > 800:
            text = ' '.join(words[:600] + words[-200:])
        
        return text.strip()

def load_and_preprocess_data(data_path):
    """Load and preprocess the resume dataset"""
    print("="*70)
    print("📊 PROCESSING DATA")
    print("="*70)
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} resumes")
    
    # Identify resume column
    resume_col = 'Resume_str' if 'Resume_str' in df.columns else 'Resume'
    print(f"✅ Using column: {resume_col}")
    
    # Clean data
    df = df.dropna(subset=[resume_col, 'Category'])
    preprocessor = ResumePreprocessor()
    df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
    df = df[df['cleaned_text'].str.len() > 50]
    print(f"✅ Cleaned {len(df)} resumes")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Category'])
    label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
    num_labels = len(label_map)
    print(f"✅ {num_labels} job categories found")
    
    # Display categories
    print("\n📋 Job Categories:")
    for idx, category in enumerate(label_encoder.classes_):
        print(f"  {idx+1:2d}. {category}")
    
    return df, label_map, num_labels

def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets"""
    texts = df['cleaned_text'].tolist()
    labels = df['label'].tolist()
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176,  # 0.176 of 85% = 15% of total
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\n✅ Data Split Complete:")
    print(f"  • Train: {len(X_train)} samples (70%)")
    print(f"  • Val:   {len(X_val)} samples (15%)")
    print(f"  • Test:  {len(X_test)} samples (15%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
