import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Language Settings
DEFAULT_LANGUAGE = 'ar'  # Arabic by default
SUPPORTED_LANGUAGES = ['ar', 'en']

# UAE Legal Resources
UAE_LEGAL_DOMAINS = [
    'https://elaws.moj.gov.ae',
    'https://www.mohre.gov.ae',
    'https://www.dm.gov.ae',
    'https://www.adjd.gov.ae',
    'https://www.dc.gov.ae'

]

# Legal Categories
LEGAL_CATEGORIES = {
    'civil': 'القانون المدني',
    'criminal': 'القانون الجنائي',
    'commercial': 'القانون التجاري',
    'labor': 'قانون العمل',
    'family': 'قانون الأسرة',
    'property': 'قانون العقارات'
}