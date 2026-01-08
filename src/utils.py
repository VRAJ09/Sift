"""
Utility functions
"""

import os
from typing import Optional

def get_api_key() -> Optional[str]:
    """Get Google Gemini API key from environment."""
    return os.getenv("GOOGLE_API_KEY")

def validate_pdf(file) -> bool:
    """Validate uploaded PDF file."""
    if file is None:
        return False
    if file.type != "application/pdf":
        return False
    return True

