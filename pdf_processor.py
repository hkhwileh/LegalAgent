import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
import arabic_reshaper
from bidi.algorithm import get_display
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
import os
from typing import List, Dict
from agents import create_judge_agent, create_advocate_agent
from crewai import Task, Crew

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for better memory management
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Initialize models with better memory management
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device_map="auto",  # Automatically choose best device
            torch_dtype=torch.float32,  # Use float32 for better memory efficiency
            batch_size=1  # Process one chunk at a time
        )
        self.progress_callback = None
        
        # Configure torch for memory efficiency
        if torch.backends.mps.is_available():  # For Mac M1/M2
            torch.backends.mps.set_per_process_memory_fraction(0.7)  # Use only 70% of available memory
        elif torch.cuda.is_available():  # For CUDA devices
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.7)
        
    def set_progress_callback(self, callback):
        """Set a callback function to report progress."""
        self.progress_callback = callback
        
    def update_progress(self, message: str, progress: float):
        """Update progress through callback if available."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF, handling both searchable and scanned PDFs with improved accuracy."""
        text = ""
        try:
            # Try to extract text directly first using PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = []
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    extracted_text.append(page_text)

            # If direct extraction yielded results, process it
            if extracted_text:
                text = "\n\n".join(extracted_text)
            else:
                # If no text was extracted, use OCR with improved settings
                images = convert_from_bytes(pdf_bytes, dpi=300)  # Higher DPI for better quality
                for image in images:
                    # Configure tesseract for better Arabic text recognition
                    custom_config = r'--oem 1 --psm 3 -l ara+eng'
                    page_text = pytesseract.image_to_string(
                        image,
                        config=custom_config,
                        lang='ara+eng'
                    )
                    if page_text.strip():
                        extracted_text.append(page_text)
                
                text = "\n\n".join(extracted_text)

            # Clean up the text
            text = self._clean_text(text)
            
            # Handle Arabic text with improved reshaping
            text = self._process_arabic_text(text)

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

        return text

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove control characters
        text = "".join(char for char in text if char.isprintable() or char in "\n\r\t")
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common OCR issues
        text = re.sub(r'(?&lt;=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'([.!?])\s*(?=[A-Z])', r'\1\n', text)
        
        # Remove empty lines and extra whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()

    def _process_arabic_text(self, text: str) -> str:
        """Process Arabic text with improved handling."""
        try:
            # Configure arabic-reshaper for better text handling
            configuration = {
                'delete_harakat': False,
                'support_ligatures': True,
                'RIAL SIGN': True
            }
            
            # Reshape Arabic text
            reshaped_text = arabic_reshaper.reshape(text, configuration=configuration)
            
            # Apply bidirectional algorithm
            text = get_display(reshaped_text)
            
            # Fix common Arabic text issues
            text = re.sub(r'([ء-ي])\s+([ء-ي])', r'\1\2', text)  # Remove spaces between Arabic letters
            text = re.sub(r'[\u200B-\u200F\u202A-\u202E]', '', text)  # Remove Unicode control characters
            
            return text
        except Exception as e:
            print(f"Warning: Error in Arabic text processing: {str(e)}")
            return text  # Return original text if processing fails

    def summarize_document(self, text: str) -> str:
        """Generate a summary of the document with improved memory management."""
        try:
            # Split text into smaller chunks
            chunks = self.text_splitter.split_text(text)
            summaries = []
            
            # Process chunks in batches to manage memory
            batch_size = 3  # Process 3 chunks at a time
            for i in range(0, len(chunks), batch_size):
                # Clear GPU/MPS memory before processing new batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    # Force garbage collection for MPS
                    import gc
                    gc.collect()
                
                batch = chunks[i:i + batch_size]
                for chunk in batch:
                    try:
                        # Generate summary with controlled length and parameters
                        summary = self.summarizer(
                            chunk,
                            max_length=130,
                            min_length=30,
                            do_sample=False,
                            num_beams=2,  # Reduced beam search for memory efficiency
                            early_stopping=True
                        )
                        summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Warning: Error summarizing chunk: {str(e)}")
                        # If summarization fails, include a portion of the original text
                        summaries.append(chunk[:200] + "...")
                
                # Update progress
                self.update_progress(
                    "جاري تلخيص المستند...",
                    min(0.3 + (i / len(chunks)) * 0.4, 0.7)
                )
            
            # Combine summaries intelligently
            final_summary = " ".join(summaries)
            
            # Clean up the final summary
            final_summary = self._clean_text(final_summary)
            final_summary = self._process_arabic_text(final_summary)
            
            return final_summary
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            # Fallback to a simple extractive summary
            return self._create_extractive_summary(text)

    def _create_extractive_summary(self, text: str, sentences_count: int = 5) -> str:
        """Create a simple extractive summary as a fallback method."""
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]\s+', text)
            
            # Remove very short sentences and clean
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            if not sentences:
                return text[:500] + "..."  # Return truncated text if no good sentences
            
            # Score sentences based on position and length
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                # Prefer sentences from the beginning and end of the document
                if i < len(sentences) * 0.3:  # First 30%
                    score += 2
                elif i > len(sentences) * 0.7:  # Last 30%
                    score += 1
                
                # Prefer medium-length sentences
                if 50 <= len(sentence) <= 200:
                    score += 1
                
                scored_sentences.append((score, sentence))
            
            # Sort by score and select top sentences
            scored_sentences.sort(reverse=True)
            selected_sentences = [s[1] for s in scored_sentences[:sentences_count]]
            
            # Sort sentences by their original order
            selected_sentences.sort(key=lambda s: sentences.index(s))
            
            # Join sentences and clean
            summary = ". ".join(selected_sentences)
            summary = self._clean_text(summary)
            summary = self._process_arabic_text(summary)
            
            return summary
            
        except Exception as e:
            print(f"Error in extractive summary: {str(e)}")
            return text[:500] + "..."  # Return truncated text as last resort
            
    def analyze_legal_issues(self, text: str) -> Dict:
        """Analyze legal issues in the document using the Judge agent."""
        judge_agent = create_judge_agent()
        
        task_description = f"""
        تحليل المستند التالي وتحديد المخالفات القانونية المحتملة وفقاً للقوانين الإماراتية:
        {text}

        يجب أن يتضمن التحليل:
        1. المخالفات القانونية المحتملة
        2. المواد القانونية ذات الصلة
        3. التوصيات للتصحيح
        """

        task = Task(
            description=task_description,
            agent=judge_agent,
            expected_output="تحليل قانوني شامل للمخالفات والتوصيات"
        )

        crew = Crew(agents=[judge_agent], tasks=[task])
        result = crew.kickoff()
        return {"legal_analysis": result}

    def map_to_uae_legislation(self, text: str) -> Dict:
        """Map document content to relevant UAE laws and regulations."""
        advocate_agent = create_advocate_agent()
        
        task_description = f"""
        تحليل المستند التالي وربطه بالقوانين والتشريعات الإماراتية ذات الصلة:
        {text}

        يجب أن يتضمن التحليل:
        1. القوانين الإماراتية ذات الصلة
        2. المواد القانونية المحددة
        3. التفسير القانوني للعلاقة
        """

        task = Task(
            description=task_description,
            agent=advocate_agent,
            expected_output="خريطة تفصيلية للقوانين والتشريعات ذات الصلة"
        )

        crew = Crew(agents=[advocate_agent], tasks=[task])
        result = crew.kickoff()
        return {"legislation_mapping": result}

    def process_document(self, pdf_bytes: bytes) -> Dict:
        """Process the document through all steps with progress tracking."""
        try:
            # Extract text from PDF
            self.update_progress("استخراج النص من المستند...", 0.1)
            text = self.extract_text_from_pdf(pdf_bytes)
            
            if not text.strip():
                raise ValueError("لم يتم العثور على نص قابل للقراءة في المستند")

            # Generate summary
            self.update_progress("إنشاء ملخص للمستند...", 0.3)
            summary = self.summarize_document(text)

            # Analyze legal issues
            self.update_progress("تحليل القضايا القانونية...", 0.5)
            legal_analysis = self.analyze_legal_issues(text)

            # Map to UAE legislation
            self.update_progress("ربط المستند بالتشريعات الإماراتية...", 0.7)
            legislation_mapping = self.map_to_uae_legislation(text)

            self.update_progress("اكتمل التحليل!", 1.0)

            return {
                "summary": summary,
                "legal_analysis": legal_analysis["legal_analysis"],
                "legislation_mapping": legislation_mapping["legislation_mapping"],
                "raw_text": text  # Include raw text for translation if needed
            }
            
        except Exception as e:
            self.update_progress(f"حدث خطأ: {str(e)}", 0)
            raise