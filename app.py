import sys
import types
import importlib
import importlib.util

# خدعة برمجية لتجاوز خطأ مكتبة imp في Python 3.12
class MockImp:
    @staticmethod
    def find_module(name, path=None):
        spec = importlib.util.find_spec(name, path)
        if spec is None:
            raise ImportError(f"No module named {name}")
        return None, spec.origin, None
    
    @staticmethod
    def load_module(name, file, pathname, description):
        return importlib.import_module(name)

sys.modules['imp'] = MockImp()
sys.modules['imp'].new_module = types.ModuleType

# -------------------------------------------------------------
# هنا تبدأ استدعاءاتك الطبيعية (لا تمسح الكود القديم، فقط اجعله أسفل هذه الخدعة)
import streamlit as st
import time
from src.router.fuzzy_controller import FuzzyController
from src.pipelines.pipeline_factory import PipelineFactory
# ... باقي كود الواجهة الذي أرسلته لك مسبقاً

import streamlit as st
import time

# -------------------------------------------------------------
# 1. استدعاء الكلاسات الحقيقية من مشروعك
# (تأكد فقط أن أسماء الكلاسات تتطابق مع ما كتبته داخل هذه الملفات)
# -------------------------------------------------------------
from src.router.fuzzy_controller import FuzzyController  # تأكد من اسم الكلاس (قد يكون FuzzyRouter)
from src.pipelines.pipeline_factory import PipelineFactory

# 2. إعدادات الصفحة
st.set_page_config(
    page_title="Dynamic Hybrid-RAG System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Dynamic Hybrid-RAG (Enterprise Edition)")
st.markdown("---")

# 3. تهيئة الذاكرة والمحرك
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"route": "N/A", "ttft": "0.0 ms", "complexity": "N/A"}

# تهيئة المحرك الذكي مرة واحدة فقط لتجنب إعادة التحميل
@st.cache_resource
def load_system():
    router = FuzzyController()
    return router

router = load_system()

# 4. الشريط الجانبي (Sidebar)
with st.sidebar:
    st.header("📊 System Metrics")
    st.markdown("Watch the Fuzzy Router in action:")
    
    st.metric(label="Active Pipeline (Router Decision)", value=st.session_state.metrics["route"])
    st.metric(label="Time-to-First-Token (TTFT)", value=st.session_state.metrics["ttft"])
    st.metric(label="Query Semantic Density", value=st.session_state.metrics["complexity"])
    
    st.markdown("---")
    st.caption("Powered by Strategy & Factory Patterns")

# 5. عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. مربع الإدخال ومعالجة السؤال
if prompt := st.chat_input("Enter your complex networking query here..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing semantic density and fetching data..."):
            
            start_time = time.time()
            
            try:
                # تشغيل الموجه الضبابي (Fuzzy Router) لاتخاذ القرار
                # افترضت هنا أن الدالة ترجع القرار ورقم التعقيد
                route_decision, complexity_score = router.evaluate(prompt) 
                
                # استدعاء المسار المناسب بناءً على القرار
                pipeline = PipelineFactory.get_pipeline(route_decision)
                
                # توليد الإجابة
                response_text = pipeline.generate_response(prompt)
                
            except Exception as e:
                response_text = f"⚠️ Error processing query: {str(e)}"
                route_decision = "Error"
                complexity_score = "N/A"
            
            # حساب الزمن
            ttft_value = round((time.time() - start_time) * 1000, 2)
            
            # تحديث المؤشرات
            st.session_state.metrics["route"] = str(route_decision)
            st.session_state.metrics["ttft"] = f"{ttft_value} ms"
            st.session_state.metrics["complexity"] = str(complexity_score)
            
            # عرض وحفظ الإجابة
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            st.rerun()