import re
from email.utils import parseaddr
from html import unescape
import pandas as pd

# Compiled regex patterns for efficiency
_URL_RE = re.compile(r'(?i)\b(?:https?://|ftp://|www\.)\S+')
_EMAIL_RE = re.compile(r'(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b')
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')

def clean_sender_receiver(text):
    """
    Parse and clean sender/receiver fields to extract name and email.
    
    Args:
        text: String like "John Doe <john@example.com>" or just "john@example.com"
        
    Returns:
        pandas Series with [name, email]
    """
    if not isinstance(text, str) or not text.strip():
        return pd.Series(["", ""])

    # Parse email using standard library
    name, email = parseaddr(text)
    name = re.sub(r'[<>"\']', '', name).strip()
    email = email.strip().lower()

    # If email not found, try regex extraction
    if not email:
        m = _EMAIL_RE.search(text)
        email = m.group(0).lower() if m else ""

    # If name not found, extract from remaining text
    if not name:
        name = re.sub(_EMAIL_RE, '', text)
        name = re.sub(r'[<>"\']', ' ', name)
        name = _WS_RE.sub(' ', name).strip()

    return pd.Series([name, email])


def clean_body(text):
    """
    Clean email body text by removing URLs, HTML, emails, and special characters.
    
    Args:
        text: Raw email body text
        
    Returns:
        Cleaned text suitable for NLP processing
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unescape HTML entities
    text = unescape(text)
    
    # Normalize obfuscated URLs (hxxp -> http)
    text = re.sub(r'(?i)\bhxxp', 'http', text)
    
    # Remove URLs (they're extracted separately)
    text = _URL_RE.sub(' ', text)
    
    # Remove HTML tags
    text = _HTML_TAG_RE.sub(' ', text)
    
    # Remove email addresses
    text = _EMAIL_RE.sub(' ', text)
    
    # Normalize whitespace characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^A-Za-z0-9\s.,!?;:\-()\'"]+', ' ', text)
    
    # Collapse multiple spaces
    text = _WS_RE.sub(' ', text).strip()

    return text
