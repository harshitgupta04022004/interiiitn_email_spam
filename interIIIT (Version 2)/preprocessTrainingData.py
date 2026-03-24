import pandas as pd 
import numpy as np
import re
from sklearn.preprocessing import normalize, LabelEncoder
import torch
from urllib.parse import urlparse
import math
import pickle
import os
from urlFeatureCreation import extract_all_url, extract_url_features
from clearnText import clean_sender_receiver, clean_body
from emailFeature import extract_email_features
from textEmbedding import doEmbedding


def processData():
    """Process training data from CSV files"""
    print("=" * 60)
    print("TRAINING DATA PREPROCESSING")
    print("=" * 60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    ceas_08 = pd.read_csv("archive/CEAS_08.csv")
    nazario = pd.read_csv("archive/Nazario_5.csv")
    nigerian = pd.read_csv("archive/Nigerian_5.csv")  
    
    # Assign spam types
    ceas_08['spam_type'] = ceas_08['label']  # 0=ham, 1=spam
    nazario['spam_type'] = 2    # Phishing
    nigerian['spam_type'] = 3   # Nigerian 419
    
    print(f"   CEAS_08: {len(ceas_08)} samples")
    print(f"   Nazario: {len(nazario)} samples (Phishing)")
    print(f"   Nigerian: {len(nigerian)} samples (Nigerian 419)")
    
    # Combine datasets
    df = pd.concat([ceas_08, nazario, nigerian], ignore_index=True)
    df.drop(columns=["label"], inplace=True)
    
    # Drop unnecessary columns if they exist
    for col in ["urls", "date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    print(f"\n2. Combined dataset shape: {df.shape}")
    print(f"   Spam type distribution:\n{df['spam_type'].value_counts().sort_index()}")
    
    # Extract URLs from body
    print("\n3. Extracting URLs from email body...")
    df['urls'] = df['body'].apply(extract_all_url)
    count_of_rows = (df['urls'].str.len() > 0).sum()
    print(f"   Emails with URLs: {count_of_rows}")
    
    # Clean sender and receiver
    print("\n4. Cleaning sender and receiver fields...")
    df[['sender_name', 'sender_email']] = df['sender'].apply(clean_sender_receiver)
    df[['receiver_name', 'receiver_email']] = df['receiver'].apply(clean_sender_receiver)
    df['body'] = df['body'].apply(clean_body)
    df.drop(columns=['sender', 'receiver', 'receiver_name'], inplace=True)
    
    # Extract URL features
    print("\n5. Extracting URL features...")
    url_feats = df['urls'].apply(extract_url_features)
    url_feats_df = pd.DataFrame(url_feats.tolist(), index=df.index)
    df = pd.concat([df, url_feats_df], axis=1)
    df.drop(columns=['urls'], inplace=True)
    print(f"   URL features shape: {url_feats_df.shape}")
    
    # Extract sender email features
    print("\n6. Extracting sender email features...")
    feats = df['sender_email'].apply(lambda x: extract_email_features(x, owner="sender"))
    feats_df = pd.DataFrame(feats.tolist(), index=df.index)
    df = pd.concat([df, feats_df], axis=1)
    df.drop(columns=['sender_email'], inplace=True)
    print(f"   Sender email features: {feats_df.shape[1]} features")
    
    # Extract receiver email features
    print("\n7. Extracting receiver email features...")
    receiver_feats = df['receiver_email'].apply(lambda x: extract_email_features(x, owner="receiver"))
    receiver_feats_df = pd.DataFrame(receiver_feats.tolist(), index=df.index)
    df = pd.concat([df, receiver_feats_df], axis=1)
    df.drop(columns=['receiver_email'], inplace=True)
    print(f"   Receiver email features: {receiver_feats_df.shape[1]} features")
    
    # Concatenate text fields
    print("\n8. Concatenating text fields...")
    def concat_all_text(sender_name, subject, body):
        sender_name = str(sender_name) if pd.notna(sender_name) else ""
        subject = str(subject) if pd.notna(subject) else ""
        body = str(body) if pd.notna(body) else ""
        return f"{sender_name} {subject} {body}".strip()
    
    df['text'] = df.apply(
        lambda row: concat_all_text(row['sender_name'], row['subject'], row['body']),
        axis=1
    )
    
    df.drop(columns=['sender_name', 'subject', 'body'], inplace=True)
    
    # Generate embeddings
    print("\n9. Generating text embeddings...")
    df = doEmbedding(df)
    
    # ============================================================
    # CRITICAL FIX: Proper TLD Encoding with LabelEncoder
    # ============================================================
    print("\n10. Encoding TLD features...")
    tld_encoders = {}
    
    for col in ['sender_email_tld', 'receiver_email_tld']:
        # Handle None/NaN values
        df[col] = df[col].fillna('unknown').astype(str)
        
        # Create and fit encoder
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        
        # Store encoder for inference
        tld_encoders[col] = encoder
        
        print(f"   {col}: {len(encoder.classes_)} unique TLDs")
        print(f"      Sample TLDs: {list(encoder.classes_[:5])}")
    
    # Save TLD encoders
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/tld_encoders.pkl', 'wb') as f:
        pickle.dump(tld_encoders, f)
    print("\n   ✓ TLD encoders saved to artifacts/tld_encoders.pkl")
    
    # Split features and target
    X, y = df.drop(columns=['spam_type']), df['spam_type']
    
    # Verify all columns are numeric
    print("\n11. Verifying data types...")
    object_cols = X.columns[X.dtypes == 'object']
    if len(object_cols) > 0:
        print(f"   ⚠️ WARNING - Columns with object dtype: {object_cols.tolist()}")
        print("   Converting to numeric...")
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    else:
        print("   ✓ All columns are numeric")
    
    # Check for boolean columns
    bool_cols = X.columns[X.dtypes == 'bool']
    if len(bool_cols) > 0:
        print(f"   ⚠️ WARNING - Boolean columns detected: {bool_cols.tolist()}")
        print("   Converting to int...")
        for col in bool_cols:
            X[col] = X[col].astype(int)
    
    print(f"\n12. Final dataset shape: X={X.shape}, y={y.shape}")
    print("=" * 60)
    return X, y


def processDataForEvaluation(sender_name, sender_email, receiver_email, subject, body):
    """
    Process a single email for prediction
    
    Args:
        sender_name: Name of the sender
        sender_email: Email address of sender
        receiver_email: Email address of receiver
        subject: Email subject line
        body: Email body text
        
    Returns:
        DataFrame with single row containing all features
    """
    print("\n" + "=" * 60)
    print("INFERENCE DATA PREPROCESSING")
    print("=" * 60)
    
    # Check if TLD encoders exist
    if not os.path.exists('artifacts/tld_encoders.pkl'):
        raise FileNotFoundError(
            "TLD encoders not found. Please train the model first by running: python train.py"
        )
    
    # Create DataFrame with single row
    df = pd.DataFrame({
        'sender_name': [sender_name],
        'sender_email': [sender_email],
        'receiver_email': [receiver_email],
        'subject': [subject],
        'body': [body]
    })
    
    print("\n1. Input data:")
    print(f"   Sender: {sender_name} <{sender_email}>")
    print(f"   Receiver: {receiver_email}")
    print(f"   Subject: {subject[:50]}...")
    print(f"   Body length: {len(body)} characters")
    
    # Extract URLs from body
    print("\n2. Extracting URLs...")
    df['urls'] = df['body'].apply(extract_all_url)
    url_count = len(df['urls'].iloc[0])
    print(f"   URLs found: {url_count}")
    
    # Clean body text
    print("\n3. Cleaning text...")
    df['body'] = df['body'].apply(clean_body)
    
    # Extract URL features
    print("\n4. Extracting URL features...")
    url_feats = df['urls'].apply(extract_url_features)
    url_feats_df = pd.DataFrame(url_feats.tolist(), index=df.index)
    df = pd.concat([df, url_feats_df], axis=1)
    df.drop(columns=['urls'], inplace=True)
    
    # Extract sender email features
    print("\n5. Extracting sender email features...")
    feats = df['sender_email'].apply(lambda x: extract_email_features(x, owner="sender"))
    feats_df = pd.DataFrame(feats.tolist(), index=df.index)
    df = pd.concat([df, feats_df], axis=1)
    df.drop(columns=['sender_email'], inplace=True)
    
    # Extract receiver email features
    print("\n6. Extracting receiver email features...")
    receiver_feats = df['receiver_email'].apply(lambda x: extract_email_features(x, owner="receiver"))
    receiver_feats_df = pd.DataFrame(receiver_feats.tolist(), index=df.index)
    df = pd.concat([df, receiver_feats_df], axis=1)
    df.drop(columns=['receiver_email'], inplace=True)
    
    # Concatenate text fields
    print("\n7. Concatenating text fields...")
    def concat_all_text(sender_name, subject, body):
        sender_name = str(sender_name) if pd.notna(sender_name) else ""
        subject = str(subject) if pd.notna(subject) else ""
        body = str(body) if pd.notna(body) else ""
        return f"{sender_name} {subject} {body}".strip()
    
    df['text'] = df.apply(
        lambda row: concat_all_text(row['sender_name'], row['subject'], row['body']),
        axis=1
    )
    
    df.drop(columns=['sender_name', 'subject', 'body'], inplace=True)
    
    # Generate embeddings
    print("\n8. Generating text embeddings...")
    df = doEmbedding(df)
    
    # ============================================================
    # CRITICAL FIX: Load and apply TLD encoders from training
    # ============================================================
    print("\n9. Encoding TLD features...")
    with open('artifacts/tld_encoders.pkl', 'rb') as f:
        tld_encoders = pickle.load(f)
    
    for col in ['sender_email_tld', 'receiver_email_tld']:
        # Handle None/NaN values
        df[col] = df[col].fillna('unknown').astype(str)
        
        # Get the TLD value
        tld_value = df[col].iloc[0]
        
        # Check if TLD was seen during training
        if tld_value in tld_encoders[col].classes_:
            df[col] = tld_encoders[col].transform(df[col])
            print(f"   {col}: '{tld_value}' → {df[col].iloc[0]} (known TLD)")
        else:
            # Unknown TLD - assign -1
            df[col] = -1
            print(f"   {col}: '{tld_value}' → -1 (unknown TLD, not seen in training)")
    
    # Return features
    X = df
    
    # Verify all columns are numeric
    print("\n10. Verifying data types...")
    object_cols = X.columns[X.dtypes == 'object']
    if len(object_cols) > 0:
        print(f"   ⚠️ WARNING - Columns with object dtype: {object_cols.tolist()}")
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Check for boolean columns
    bool_cols = X.columns[X.dtypes == 'bool']
    if len(bool_cols) > 0:
        print(f"   ⚠️ WARNING - Boolean columns: {bool_cols.tolist()}")
        for col in bool_cols:
            X[col] = X[col].astype(int)
    
    print(f"\n11. Final features shape: {X.shape}")
    print("=" * 60)
    return X
