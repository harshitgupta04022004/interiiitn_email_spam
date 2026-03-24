import re
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import math

_SUSPICIOUS_KEYWORDS = {
    "login", "secure", "confirm", "verify", "account", "signin", "bank",
    "update", "password", "wp-admin", "enter", "redirect", "auth"
}

_URL_SHORTENERS = {
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "buff.ly",
    "is.gd", "tiny.cc", "rebrand.ly", "shorturl.at", "adf.ly", "shorte.st",
    "rb.gy", "youtu.be"
}

_IPv4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_IPv6_RE = re.compile(r"^\[?[A-F0-9:]+\]?$", re.I) 


def extract_all_url(text):
    """
    Extract all URLs from text using multiple patterns.
    
    Args:
        text: String containing potential URLs
        
    Returns:
        List of unique URLs found in the text
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Normalize hxxp to http for better URL detection
    text = re.sub(r'hxxp', 'http', text, flags=re.IGNORECASE)
    
    # Multiple patterns to catch different URL formats
    patterns = [
        r'href=[\'"]?([^\'" >]+)',   # HTML href attributes
        r'(?i)\b((?:https?://|http://|ftp://|ftps://|www\.)[^\s<>"\'\)\]]+)',  # Standard URLs
        r'(?i)\b((?:[a-z0-9\-]+\.)+[a-z]{2,}(?:/[^\s<>"\'\)\]]*)?)'  # Domain-like patterns
    ]
    
    found = []
    for pat in patterns:
        for match in re.findall(pat, text):
            u = match.strip()
            u = u.rstrip('.,;:)"\'<>]')
            
            # Skip email addresses
            if re.fullmatch(r'[\w\.-]+@[\w\.-]+', u):
                continue
            
            # Add http:// to www. URLs
            if u.lower().startswith('www.'):
                u = 'http://' + u
            
            # Skip very short strings
            if len(u) < 6:
                continue
            
            found.append(u)
    
    # Remove duplicates while preserving order
    seen = set()
    results = []
    for u in found:
        if u not in seen:
            seen.add(u)
            results.append(u)
    
    return results


def extract_numerical_feature_from_one_url(url_text):
    """
    Extract numerical features from a single URL.
    
    Returns:
        Tuple of (length, special_char_count, digits_to_letters_ratio, 
                 suspicious_keyword_count, redirection_count, hyphen_count_in_domain)
    """
    if not isinstance(url_text, str) or url_text.strip() == "":
        return (0, 0, 0.0, 0, 0, 0)
    
    u = url_text.strip()
    length = len(u)
    
    # Count special characters
    special_chars = r".-/?&=%_+#!:@"
    special_char_count = sum(u.count(ch) for ch in special_chars)
    
    # Digit to letter ratio
    digits = sum(c.isdigit() for c in u)
    letters = sum(c.isalpha() for c in u)
    digits_to_letters_ratio = digits / (letters + 1e-9)
    
    # Suspicious keyword count
    low = u.lower()
    suspicious_keyword_count = sum(low.count(k) for k in _SUSPICIOUS_KEYWORDS)
    
    # Redirection count (multiple //)
    double_slash_count = low.count("//")
    redirection_count = max(0, double_slash_count - 1)
    
    # Hyphen count in domain
    try:
        parsed = urlparse(u if "://" in u else "http://" + u)
        domain = parsed.netloc.lower()
    except Exception:
        domain = ""
    
    hyphen_count_in_domain = domain.count('-')
    
    return (
        length,
        special_char_count,
        digits_to_letters_ratio,
        suspicious_keyword_count,
        redirection_count,
        hyphen_count_in_domain
    )


def extract_patterns_feature_from_one_url(url_text):
    """
    Extract pattern-based features from a single URL.
    
    Returns:
        Tuple of (presence_of_ip_address, url_shorteningservices)
        Both are boolean indicators
    """
    if not isinstance(url_text, str) or url_text.strip() == "":
        return (False, False)
    
    u = url_text.strip()
    
    # Extract domain
    try:
        parsed = urlparse(u if "://" in u else "http://" + u)
        domain = parsed.netloc.lower()
        domain = domain.split(":")[0]  # Remove port if present
    except Exception:
        domain = ""
    
    # Check for IP address
    is_ipv4 = bool(_IPv4_RE.match(domain))
    is_ipv6 = False
    if ":" in domain:
        is_ipv6 = bool(_IPv6_RE.match(domain))
    presence_of_ip_address = is_ipv4 or is_ipv6
    
    # Check for URL shortening service
    domain_only = domain.lower()
    url_shorteningservices = False
    
    if domain_only in _URL_SHORTENERS:
        url_shorteningservices = True
    else:
        # Additional heuristic: short domain name + path might be a shortener
        if domain_only:
            labels = domain_only.split('.')
            if len(labels) >= 2:
                name_label = labels[-2]
            else:
                name_label = labels[0]
            
            # Very short domain names with paths are often shorteners
            if len(name_label) <= 4:
                parsed = urlparse(u if "://" in u else "http://" + u)
                if parsed.path and parsed.path.strip() not in ("/", ""):
                    url_shorteningservices = True
    
    return (presence_of_ip_address, url_shorteningservices)


def extract_url_features(urls_list):
    """
    Extract aggregated URL features from a list of URLs.
    
    Args:
        urls_list: Either a Python list of URL strings, or a single URL string
        
    Returns:
        Tuple of 8 features:
        (avg_length, avg_special_char_count, avg_digits_to_letters_ratio,
         avg_suspicious_keyword_count, avg_redirection_count, avg_hyphen_count_in_domain,
         presence_of_ip_address_any, url_shorteningservice_any)
         
        All values are numeric (int or float). Last two are int (0/1) not bool.
    """
    # Handle different input types
    if urls_list is None:
        urls = []
    elif isinstance(urls_list, str):
        urls = [urls_list]
    else:
        try:
            urls = list(urls_list)
        except Exception:
            urls = [str(urls_list)]

    # No URLs case
    if len(urls) == 0:
        return (0, 0, 0.0, 0, 0, 0, 0, 0)  # FIXED: Last two are 0 (int), not False (bool)

    # Accumulate features
    accum = [0.0] * 6
    presence_ip = False
    shortening = False
    count = 0
    
    for u in urls:
        # Skip invalid entries
        if u is None:
            continue
        if isinstance(u, float) and math.isnan(u):
            continue
        if isinstance(u, str) and u.strip() == "":
            continue
        
        # Extract features
        num_feats = extract_numerical_feature_from_one_url(u)
        pat_feats = extract_patterns_feature_from_one_url(u)
        
        # Accumulate numerical features
        for i in range(6):
            accum[i] += float(num_feats[i])
        
        # Accumulate pattern features (OR logic)
        presence_ip = presence_ip or bool(pat_feats[0])
        shortening = shortening or bool(pat_feats[1])
        count += 1
    
    # No valid URLs case
    if count == 0:
        return (0, 0, 0.0, 0, 0, 0, 0, 0)  # FIXED: Last two are 0 (int), not False (bool)
    
    # Calculate averages
    avg_feats = [acc / count for acc in accum]

    # CRITICAL FIX: Return integers (0/1) instead of booleans for XGBoost
    return (
        avg_feats[0],                     # avg_length
        avg_feats[1],                     # avg_special_char_count
        avg_feats[2],                     # avg_digits_to_letters_ratio
        avg_feats[3],                     # avg_suspicious_keyword_count
        avg_feats[4],                     # avg_redirection_count
        avg_feats[5],                     # avg_hyphen_count_in_domain
        int(presence_ip),                 # FIXED: Convert bool to int (0 or 1)
        int(shortening)                   # FIXED: Convert bool to int (0 or 1)
    )
