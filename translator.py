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
        """Translate text from source language to target language with improved handling."""
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
        
        try:
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Split text into manageable chunks
            chunks = self._split_text_into_chunks(text)
            translated_chunks = []
            
            for chunk in chunks:
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Tokenize with improved settings
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True
                )
                
                # Generate translation with improved settings
                with torch.no_grad():
                    translated = model.generate(
                        **inputs,
                        num_beams=2,  # Reduced for memory efficiency
                        length_penalty=0.6,
                        max_length=512,
                        min_length=0,
                        early_stopping=True
                    )
                
                # Decode the translation
                result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                translated_chunks.append(result)
            
            # Combine chunks
            final_translation = ' '.join(translated_chunks)
            
            # Post-process translation
            final_translation = self._post_process_translation(final_translation, target_lang)
            
            return final_translation
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
        
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
        
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 450) -> list:
        """Split text into manageable chunks for translation."""
        # First try to split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            # If a single paragraph is too long, split it by sentences
            if len(para) > max_chunk_size:
                sentences = re.split(r'([.!?])\s+', para)
                i = 0
                while i < len(sentences):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]  # Add back the punctuation
                        i += 2
                    else:
                        i += 1
                        
                    if current_length + len(sentence) > max_chunk_size:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            else:
                if current_length + len(para) > max_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(para)
                current_length += len(para)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _post_process_translation(self, text: str, target_lang: str) -> str:
        """Post-process translated text based on target language."""
        if target_lang.lower() in ['arabic', 'ar']:
            # Fix Arabic-specific issues
            text = re.sub(r'([ء-ي])\s+([ء-ي])', r'\1\2', text)  # Remove spaces between Arabic letters
            text = re.sub(r'[\u200B-\u200F\u202A-\u202E]', '', text)  # Remove Unicode control characters
            
            # Fix common Arabic punctuation issues
            text = text.replace('،,', '،')
            text = text.replace('.,', '.')
            text = text.replace('؟?', '؟')
            text = text.replace('!!', '!')
            
            # Ensure proper spacing around numbers and Latin text
            text = re.sub(r'([0-9])([ء-ي])', r'\1 \2', text)
            text = re.sub(r'([ء-ي])([0-9])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z])([ء-ي])', r'\1 \2', text)
            text = re.sub(r'([ء-ي])([a-zA-Z])', r'\1 \2', text)
            
        elif target_lang.lower() in ['english', 'en']:
            # Fix English-specific issues
            text = re.sub(r'\s+([.,!?])', r'\1', text)  # Fix spacing before punctuation
            text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)  # Fix spacing after punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalize spaces
            text = text.replace(' ,', ',')
            text = text.replace(' .', '.')
            
            # Capitalize first letter of sentences
            text = '. '.join(s.capitalize() for s in text.split('. '))
            
        return text.strip()

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