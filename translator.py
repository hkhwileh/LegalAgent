from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
from langdetect import detect
import re

class Translator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.language_codes = {
            'arabic': 'ar',
            'english': 'en',
            'chinese': 'zh',
            'hindi': 'hi',
            'urdu': 'ur'
        }
        
        # Initialize models for each language pair
        self._load_model('en', 'ar')  # English to Arabic
        self._load_model('ar', 'en')  # Arabic to English
        # Add other language pairs as needed
        
    def _load_model(self, src_lang, tgt_lang):
        """Load translation model for a specific language pair."""
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        key = f'{src_lang}-{tgt_lang}'
        
        if key not in self.models:
            try:
                self.tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
                self.models[key] = MarianMTModel.from_pretrained(model_name)
            except Exception as e:
                print(f"Error loading model for {key}: {str(e)}")
                
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source language to target language."""
        src_code = self.language_codes.get(source_lang.lower())
        tgt_code = self.language_codes.get(target_lang.lower())
        
        if not src_code or not tgt_code:
            raise ValueError("Unsupported language")
            
        key = f'{src_code}-{tgt_code}'
        
        if key not in self.models:
            self._load_model(src_code, tgt_code)
            
        if key not in self.models:
            raise ValueError(f"Translation model not available for {source_lang} to {target_lang}")
            
        tokenizer = self.tokenizers[key]
        model = self.models[key]
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate translation
        with torch.no_grad():
            translated = model.generate(**inputs)
            
        # Decode the translation
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        return result
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            # Clean text for better detection
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            detected = detect(cleaned_text)
            
            # Map detected language code to our supported languages
            lang_code_map = {
                'ar': 'arabic',
                'en': 'english',
                'zh': 'chinese',
                'hi': 'hindi',
                'ur': 'urdu'
            }
            
            return lang_code_map.get(detected, 'english')  # Default to English if unknown
        except:
            return 'english'  # Default to English if detection fails
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before translation."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters that might interfere with translation
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
        return text
    
    def get_supported_languages(self):
        """Return list of supported languages."""
        return list(self.language_codes.keys())
        
    def get_language_name(self, code: str) -> str:
        """Get the display name for a language code."""
        names = {
            'ar': 'العربية',
            'en': 'English',
            'zh': '中文',
            'hi': 'हिंदी',
            'ur': 'اردو'
        }
        return names.get(code, code)