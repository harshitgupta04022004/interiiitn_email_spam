import re
import math
import ipaddress
from collections import Counter

FREE_PROVIDERS = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "protonmail.com", "mail.com", "zoho.com", "aol.com"
})

SUSPICIOUS_KEYWORDS = frozenset({
    "admin", "support", "secure", "login", "verify",
    "bank", "update", "noreply", "service", "help"
})

_EMAIL_RE = re.compile(r"^(?P<local>[^@\s]+)@(?P<domain>[^@\s]+\.[^@\s]+)$")

def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string"""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def _is_ip(value: str) -> bool:
    """Check if string is a valid IP address"""
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False

def extract_email_features(email: str, owner: str = "sender"):
    """
    Extract comprehensive features from an email address.
    
    Args:
        email: Email address string
        owner: Prefix for feature names ("sender" or "receiver")
        
    Returns:
        Dictionary of numeric features (all values are int or float)
    """
    p = f"{owner}_email_"
    
    # Initialize all features with default values
    f = {
        p + "len_email": 0,
        p + "len_local": 0,
        p + "len_domain": 0,
        p + "num_dots_domain": 0,
        p + "num_subdomains": 0,
        p + "num_special_local": 0,
        p + "num_digits_local": 0,
        p + "num_digits_domain": 0,
        p + "ratio_digits_letters_local": 0.0,
        p + "has_plus": 0,                      # FIXED: Changed from bool to int
        p + "has_dot_in_local": 0,              # FIXED: Changed from bool to int
        p + "domain_is_ip": 0,                  # FIXED: Changed from bool to int
        p + "is_free_provider": 0,              # FIXED: Changed from bool to int
        p + "has_suspicious_keyword_local": 0,  # FIXED: Changed from bool to int
        p + "local_entropy": 0.0,
        p + "domain_entropy": 0.0,
        p + "tld": None,
    }

    # Handle invalid input
    if not isinstance(email, str):
        return f

    email = email.strip()
    if not email:
        return f

    # Basic length feature
    f[p + "len_email"] = len(email)

    # Parse email into local and domain parts
    m = _EMAIL_RE.match(email)
    if m:
        local = m.group("local")
        domain = m.group("domain").lower().split(":", 1)[0].strip()
    else:
        local = email
        domain = ""

    domain_parts = [x for x in domain.split(".") if x]
    local_lower = local.lower()

    # Length features
    f[p + "len_local"] = len(local)
    f[p + "len_domain"] = len(domain)
    f[p + "num_dots_domain"] = domain.count(".")
    f[p + "num_subdomains"] = max(0, len(domain_parts) - 2) if len(domain_parts) >= 3 else 0

    # Character analysis features
    f[p + "num_special_local"] = sum(1 for c in local if not c.isalnum())
    digits_local = sum(c.isdigit() for c in local)
    letters_local = sum(c.isalpha() for c in local)
    f[p + "num_digits_local"] = digits_local
    f[p + "num_digits_domain"] = sum(c.isdigit() for c in domain)
    f[p + "ratio_digits_letters_local"] = digits_local / letters_local if letters_local else 0.0

    # Pattern features - CRITICAL FIX: Convert boolean to int
    f[p + "has_plus"] = int("+" in local)
    f[p + "has_dot_in_local"] = int("." in local)
    f[p + "has_suspicious_keyword_local"] = int(any(k in local_lower for k in SUSPICIOUS_KEYWORDS))

    # Entropy features
    f[p + "local_entropy"] = shannon_entropy(local)
    f[p + "domain_entropy"] = shannon_entropy(domain)

    # Domain type features - CRITICAL FIX: Convert boolean to int
    f[p + "domain_is_ip"] = int(_is_ip(domain) if domain else False)
    f[p + "is_free_provider"] = int(domain in FREE_PROVIDERS)

    # TLD extraction (kept as string/None for later encoding)
    f[p + "tld"] = domain_parts[-1] if domain_parts else None
    
    return f
