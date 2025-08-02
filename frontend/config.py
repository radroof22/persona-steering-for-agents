import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Clustered Prompt Personalization Demo",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color Palette
COLORS = {
    "background": "#FAFAFA",
    "primary": "#007ACC",
    "secondary": "#00BFA5",
    "text": "#333333"
} 