import streamlit as st
from agents import create_judge_agent, create_advocate_agent, create_consultant_agent
from crewai import Task, Crew
from utils import is_arabic, format_legal_response
from config import LEGAL_CATEGORIES, DEFAULT_LANGUAGE

st.set_page_config(page_title="المساعد القانوني الإماراتي", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("المساعد القانوني الإماراتي")
st.write("احصل على المساعدة القانونية من خبراء قانونيين إماراتيين مدعومين بالذكاء الاصطناعي")

# Add imports
from pdf_processor import PDFProcessor
from document_exporter import DocumentExporter
from translator import Translator

# Initialize components
if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if 'document_exporter' not in st.session_state:
    st.session_state.document_exporter = DocumentExporter()
if 'translator' not in st.session_state:
    st.session_state.translator = Translator()

# Create a new tab for PDF upload
tab1, tab2, tab3, tab4 = st.tabs(["تحليل المستندات", "القاضي", "المحامي", "المستشار"])

# PDF Upload Tab
with tab1:
    st.header("تحليل المستندات القانونية")
    
    # Add service selection toggle
    service_type = st.radio(
        "اختر نوع الخدمة / Select Service",
        ["تلخيص وتحليل المستند", "ترجمة المستند"],
        horizontal=True
    )
    
    if service_type == "ترجمة المستند":
        target_language = st.selectbox(
            "اختر لغة الترجمة / Select Target Language",
            ["العربية", "English", "中文", "हिंदी", "اردو"],
            index=1
        )
    
    uploaded_file = st.file_uploader("قم بتحميل ملف PDF للتحليل", type=['pdf'])
    
    if uploaded_file is not None:
        # Check file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
        if file_size > 20:  # 20MB limit
            st.error("حجم الملف كبير جداً. الحد الأقصى المسموح به هو 20 ميجابايت.")
            st.stop()

        if service_type == "تلخيص وتحليل المستند":
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)

            st.session_state.pdf_processor.set_progress_callback(update_progress)

            try:
                # Process the uploaded PDF
                results = st.session_state.pdf_processor.process_document(uploaded_file.read())
                
                # Display results in collapsible sections
                with st.expander("ملخص المستند", expanded=True):
                    st.write(results["summary"])
                
                with st.expander("تحليل المخالفات القانونية", expanded=True):
                    st.markdown(results["legal_analysis"], unsafe_allow_html=True)
                
                with st.expander("الخريطة التشريعية", expanded=True):
                    st.markdown(results["legislation_mapping"], unsafe_allow_html=True)
                
                # Add export buttons in a container
                st.markdown("### تحميل التحليل")
                export_container = st.container()
                
                col1, col2 = export_container.columns(2)
                with col1:
                    pdf_button = st.download_button(
                        label="تحميل كملف PDF",
                        data=st.session_state.document_exporter.export_to_pdf(results),
                        file_name="legal_analysis.pdf",
                        mime="application/pdf",
                        key="pdf_download"
                    )
                
                with col2:
                    word_button = st.download_button(
                        label="تحميل كملف Word",
                        data=st.session_state.document_exporter.export_to_word(results),
                        file_name="legal_analysis.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="word_download"
                    )
                
            except ValueError as ve:
                st.error(f"خطأ في المدخلات: {str(ve)}")
            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {str(e)}")
                st.error("يرجى المحاولة مرة أخرى أو الاتصال بالدعم الفني")
            finally:
                # Clear progress bar and status
                progress_bar.empty()
                status_text.empty()
        
        else:  # Translation service
            with st.spinner("جاري تحليل المستند..."):
                try:
                    # Extract text from PDF
                    text = st.session_state.pdf_processor.extract_text_from_pdf(uploaded_file.read())
                    
                    if not text.strip():
                        st.error("لم يتم العثور على نص قابل للقراءة في المستند")
                        st.stop()
                    
                    # Detect source language
                    source_lang = st.session_state.translator.detect_language(text)
                    st.info(f"تم اكتشاف لغة المستند: {st.session_state.translator.get_language_name(source_lang)}")
                    
                    # Map language names to codes
                    lang_map = {
                        "العربية": "arabic",
                        "English": "english",
                        "中文": "chinese",
                        "हिंदी": "hindi",
                        "اردو": "urdu"
                    }
                    
                    target_lang = lang_map[target_language]
                    
                    # Check if source and target are the same
                    if source_lang == target_lang:
                        st.warning("لغة المصدر ولغة الهدف متطابقتان. يرجى اختيار لغة مختلفة للترجمة.")
                        st.stop()
                    
                    with st.spinner("جاري الترجمة..."):
                        # Preprocess and translate the text
                        processed_text = st.session_state.translator.preprocess_text(text)
                        translated_text = st.session_state.translator.translate(
                            processed_text,
                            source_lang,
                            target_lang
                        )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("النص الأصلي / Original Text")
                        st.text_area("", value=text, height=300, key="original_text")
                    
                    with col2:
                        st.subheader("النص المترجم / Translated Text")
                        st.text_area("", value=translated_text, height=300, key="translated_text")
                    
                    # Add download buttons
                    st.markdown("### تحميل الترجمة")
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        st.download_button(
                            label="تحميل النص المترجم",
                            data=translated_text.encode(),
                            file_name=f"translated_document.txt",
                            mime="text/plain",
                            key="translation_download"
                        )
                    
                    with download_col2:
                        # Create a simple HTML file with both texts
                        html_content = f"""
                        <html dir="auto">
                        <head>
                            <meta charset="UTF-8">
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                .text-container {{ margin-bottom: 20px; }}
                                h2 {{ color: #2c3e50; }}
                            </style>
                        </head>
                        <body>
                            <div class="text-container">
                                <h2>Original Text</h2>
                                <p>{text}</p>
                            </div>
                            <div class="text-container">
                                <h2>Translated Text</h2>
                                <p>{translated_text}</p>
                            </div>
                        </body>
                        </html>
                        """
                        
                        st.download_button(
                            label="تحميل النصين معاً (HTML)",
                            data=html_content.encode(),
                            file_name="translation_with_original.html",
                            mime="text/html",
                            key="html_download"
                        )
                    
                except ValueError as ve:
                    st.error(f"خطأ في المدخلات: {str(ve)}")
                except Exception as e:
                    st.error(f"حدث خطأ غير متوقع: {str(e)}")
                    st.error("يرجى المحاولة مرة أخرى أو الاتصال بالدعم الفني")

# Language selector
language = st.sidebar.selectbox(
    "اختر اللغة / Select Language",
    ["العربية", "English"],
    index=0
)

# Legal category selector
selected_category = st.sidebar.selectbox(
    "اختر الفئة القانونية / Select Legal Category",
    list(LEGAL_CATEGORIES.values()),
    index=0
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create tabs for different agents
tab1, tab2, tab3 = st.tabs(["القاضي", "المحامي", "المستشار"])

def get_agent_response(agent, query, category):
    # Prepare the task with context
    task_description = f"""
    تحليل والرد على الاستفسار التالي في مجال {category}:
    {query}
    
    يجب أن يكون الرد:
    1. مستنداً إلى القانون الإماراتي
    2. مدعوماً بالمراجع القانونية
    3. واضحاً ومفهوماً
    4. متوافقاً مع أحدث التشريعات
    """
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output="تحليل قانوني ورد بناءً على القانون الإماراتي"

    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task]
    )
    
    result = crew.kickoff()
    return format_legal_response(result, 'ar' if is_arabic(query) else 'en')

# Judge Tab
with tab2:
    st.header("استشارة القاضي الإماراتي")
    judge_query = st.text_area("اكتب سؤالك القانوني للقاضي:", key="judge_input", placeholder="أدخل النص هنا...")
    st.markdown(
        """
        <style>
        .element-container textarea {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("الحصول على رأي القاضي", key="judge_button"):
        if judge_query:
            with st.spinner("القاضي يحلل قضيتك..."):
                judge_agent = create_judge_agent()
                response = get_agent_response(judge_agent, judge_query, selected_category)
                st.session_state.chat_history.append(("القاضي", judge_query, response))
                st.write("رد القاضي:")
                st.markdown(response, unsafe_allow_html=True)

# Advocate Tab
with tab3:
    st.header("استشارة المحامي الإماراتي")
    advocate_query = st.text_area("اكتب سؤالك القانوني للمحامي:", key="advocate_input", placeholder="أدخل النص هنا...")
    st.markdown(
        """
        <style>
        .element-container textarea {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("الحصول على رأي المحامي", key="advocate_button"):
        if advocate_query:
            with st.spinner("المحامي يحلل قضيتك..."):
                advocate_agent = create_advocate_agent()
                response = get_agent_response(advocate_agent, advocate_query, selected_category)
                st.session_state.chat_history.append(("المحامي", advocate_query, response))
                st.write("رد المحامي:")
                st.markdown(response, unsafe_allow_html=True)

# Consultant Tab
with tab4:
    st.header("استشارة المستشار القضائي الإماراتي")
    consultant_query = st.text_area("اكتب سؤالك القانوني للمستشار:", key="consultant_input", placeholder="أدخل النص هنا...")
    st.markdown(
        """
        <style>
        .element-container textarea {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
