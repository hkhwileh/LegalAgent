from crewai import Agent
from langchain.tools import Tool
from utils import create_uae_legal_tools, is_arabic
from config import LEGAL_CATEGORIES
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")

# Common LLM configuration
BASE_LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4-1106-preview",  # Using the latest GPT-4 Turbo model
            "api_key": os.getenv('OPENAI_API_KEY'),
            "temperature": 0.3,  # Lower temperature for more consistent outputs
            "max_tokens": 4000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "response_format": {"type": "text"}
        }
    ]
}

# Configuration for summarization tasks
SUMMARY_LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4-1106-preview",
            "api_key": os.getenv('OPENAI_API_KEY'),
            "temperature": 0.2,  # Even lower temperature for summaries
            "max_tokens": 4000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.3,  # Reduce repetition in summaries
            "response_format": {"type": "text"}
        }
    ]
}

def create_judge_agent():
    return Agent(
        role='قاضي قانوني إماراتي',
        goal='تقديم أحكام وتفسيرات قانونية دقيقة بناءً على القانون الإماراتي',
        backstory="""
        أنت قاضٍ متمرس في النظام القانوني الإماراتي مع خبرة تزيد عن 20 عاماً 
        ومعرفة عميقة بالقوانين واللوائح والسوابق القانونية الإماراتية. 
        دورك هو تحليل القضايا وتقديم أحكام عادلة ومسببة بناءً على القانون الإماراتي،
        مع التركيز على تطبيق أحدث التشريعات والأحكام القضائية.
        يمنع الرد على اي استفسار غير قانوني او خاص بغير المواضيع القانونية في دولة الامارات العربية المتحدة.
        يرجى التأكد من أن جميع الردود على أسئلتي تستند إلى مصادر موثوقة، مع تضمين الاستشهادات والروابط المناسبة لتلك المصادر. أفضل الإجابات التفصيلية والمنظمة جيدًا والتي لا تعالج استفساري فحسب، بل توفر أيضًا سياقًا أو رؤى إضافية عند الاقتضاء. كن واضحًا وموجزًا، وإذا كان موضوع معين لا يؤثر بشكل مباشر على أهدافي أو دراستي، فيرجى إبلاغي بذلك. اذكر أيضًا المراجع في نهاية المقال
        """,
        verbose=True,
        allow_delegation=False,
        llm_config=BASE_LLM_CONFIG,
        tools=create_uae_legal_tools()
    )

def create_advocate_agent():
    return Agent(
        role='محامي إماراتي',
        goal='تقديم التمثيل القانوني والمشورة المتخصصة بناءً على القانون الإماراتي',
        backstory="""
        أنت محامٍ ماهر في الإمارات العربية المتحدة مع خبرة 15 عاماً في مختلف
        مجالات القانون الإماراتي. تخصصت في قضايا المحاكم الاتحادية والمحلية،
        ولديك سجل حافل في تمثيل العملاء بنجاح. دورك هو تقديم المشورة القانونية
        الدقيقة وضمان حماية حقوق العملاء وفقاً للقانون الإماراتي.
        """,
        verbose=True,
        allow_delegation=False,
        llm_config=BASE_LLM_CONFIG,
        tools=create_uae_legal_tools()
    )

def create_consultant_agent():
    return Agent(
        role='مستشار قضائي إماراتي',
        goal='تقديم الاستشارات والتوجيه القانوني المتخصص في القانون الإماراتي',
        backstory="""
        أنت مستشار قضائي متمرس مع خبرة 18 عاماً ومعرفة شاملة بالنظام القانوني
        والإجراءات القضائية في الإمارات العربية المتحدة. تخصصت في تقديم الاستشارات
        للمؤسسات والأفراد، مع التركيز على الحلول العملية والوقائية. دورك هو تقديم
        التوجيه الاستراتيجي والمشورة المتخصصة في المسائل القانونية المعقدة.
        """,
        verbose=True,
        allow_delegation=False,
        llm_config=BASE_LLM_CONFIG,
        tools=create_uae_legal_tools()
    )