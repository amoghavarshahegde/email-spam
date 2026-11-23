import re
import html

def clean_text(text: str) -> str:
    """
    Cleans raw text message:
    1. Lowercases text
    2. Decodes HTML entities and strips HTML tags
    3. Replaces URLs with ' url '
    4. Replaces email addresses with ' email '
    5. Replaces numbers with ' num '
    6. Strips punctuation and symbols
    7. Collapses consecutive whitespace characters
    """
    if text is None:
        return ""
    
    t = str(text).lower()

    # Unescape HTML entities (e.g. &amp; -> &) and remove HTML tags
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)

    # Replace email addresses first
    t = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " email ", t)

    # Replace URLs
    t = re.sub(r"https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|co|io|uk|de)\b\S*", " url ", t)

    # Replace numbers/digits
    t = re.sub(r"\b\d+[\d,.]*\b|\d+", " num ", t)

    # Strip symbols and punctuation (keep letters, numbers already converted to num, and spaces)
    t = re.sub(r"[^\w\s]", " ", t)

    # Collapse multiple whitespace characters into a single space and strip
    t = re.sub(r"\s+", " ", t).strip()

    return t

if __name__ == "__main__":
    sample = "<b>WINNER!</b> Claim $500 cash now at http://win-now.com or email win@cash.org! Call 12345!"
    print("Sample input:", sample)
    print("Cleaned text:", clean_text(sample))
