from typing import List, Dict
import re
from langchain.tools import Tool
from config import UAE_LEGAL_DOMAINS

def is_arabic(text: str) -> bool:
    """Check if the text contains Arabic characters."""
    arabic_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def create_uae_legal_tools() -> List[Tool]:
    """Create tools for UAE legal research."""
    tools = [
        Tool(
            name="UAE Legal Database Search",
            func=lambda q: search_uae_legal_database(q),
            description="Search UAE legal databases for laws, regulations, and precedents"
        ),
        Tool(
            name="Arabic Legal Term Translation",
            func=lambda q: translate_legal_term(q),
            description="Translate legal terms between Arabic and English"
        ),
        Tool(
            name="UAE Case Law Search",
            func=lambda q: search_uae_case_law(q),
            description="Search UAE case law and legal precedents"
        )
    ]
    return tools

def search_uae_legal_database(query: str) -> str:
    """Simulate searching UAE legal databases."""
    # In a real implementation, this would connect to actual UAE legal databases
    return f"Found relevant UAE legal information for: {query}"

def translate_legal_term(term: str) -> str:
    """Simulate legal term translation."""
    # In a real implementation, this would use a legal terms dictionary
    return f"Translation for: {term}"

def search_uae_case_law(query: str) -> str:
    """Simulate searching UAE case law."""
    # In a real implementation, this would search actual UAE case law databases
    return f"Found relevant UAE case law for: {query}"

def format_legal_response(response: str, language: str = 'ar') -> str:
    """Format legal responses with proper styling and language direction."""
    if language == 'ar':
        return f'<div dir="rtl">{response}</div>'
    return response