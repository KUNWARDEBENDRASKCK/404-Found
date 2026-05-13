from groq import Groq
import requests
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        # Get only plain text, remove HTML tags
        text = response.text
        clean = ""
        inside_tag = False
        for char in text:
            if char == "<":
                inside_tag = True
            elif char == ">":
                inside_tag = False
            elif not inside_tag:
                clean += char
        # Return first 500 characters
        return clean[:500].strip()
    except:
        return None

def analyze_text(text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this news and tell me if it is REAL or FAKE.

News: {text}

Reply in this exact format only:
VERDICT: REAL or FAKE
SCORE: (0-100, where 100 = definitely real)
REASON: (one short sentence)"""
            }
        ]
    )
    return response.choices[0].message.content

def analyze_url(url):
    text = get_text_from_url(url)
    if text:
        return analyze_text(text)
    else:
        return "Could not fetch article from URL"

def analyze(input_text_or_url):
    if input_text_or_url.startswith("http"):
        return analyze_url(input_text_or_url)
    else:
        return analyze_text(input_text_or_url)
    # ============================================
# FINAL CLEAN OUTPUT FOR BACKEND TEAMMATE
# ============================================

def analyze_final(input_text_or_url):
    raw = analyze(input_text_or_url)
    
    # Parse the response into a clean dictionary
    lines = raw.strip().split("\n")
    result = {
        "verdict": "UNKNOWN",
        "credibility_score": 50,
        "reason": "Could not analyze"
    }
    
    for line in lines:
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("SCORE:"):
            try:
                result["credibility_score"] = int(line.replace("SCORE:", "").strip())
            except:
                result["credibility_score"] = 50
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
    
    return result
