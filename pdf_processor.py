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
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # Initialize models
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """Set a callback function to report progress."""
        self.progress_callback = callback
        
    def update_progress(self, message: str, progress: float):
        """Update progress through callback if available."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF, handling both searchable and scanned PDFs."""
        text = ""
        try:
            # Try to extract text directly first
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text()

            # If no text was extracted, try OCR
            if not text.strip():
                images = convert_from_bytes(pdf_bytes)
                for image in images:
                    text += pytesseract.image_to_string(image, lang='ara+eng') + "\n"

            # Handle Arabic text
            reshaped_text = arabic_reshaper.reshape(text)
            text = get_display(reshaped_text)

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

        return text

    def summarize_document(self, text: str) -> str:
        """Generate a summary of the document."""
        chunks = self.text_splitter.split_text(text)
        summaries = []

        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        return " ".join(summaries)

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