import streamlit as st
import os
import json 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
import time 

# استيراد أدوات البحث
from langchain_community.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper # أداة البحث في جوجل Serper

# --- استيراد مكتبة dotenv لتحميل المتغيرات من ملف .env ---
from dotenv import load_dotenv

# --- تحميل المتغيرات من ملف .env (يجب أن تكون في بداية الكود) ---
load_dotenv()


google_api_key = st.secrets["GOOGLE_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]


# --- تعريف عتبة المسافة لأسئلة الـ FAQ ---
FAQ_DISTANCE_THRESHOLD = 0.2 


# --- إعدادات صفحة Streamlit (يجب أن تكون في البداية) ---
st.set_page_config(
    page_title="شات بوت جامعة الشام - مساعدك الذكي", 
    page_icon="🎓", 
    layout="centered", 
    initial_sidebar_state="expanded" 
)

# --- Custom CSS لتصميم الواجهة (مستوحى من Chat Copilot) ---
st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@200..1000&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'); /* أضف هذا السطر لاستيراد Font Awesome */

    /* ... (بقية كود CSS الخاص بك) ...
    body, html, .st-emotion-cache-ch5fby, h1, h2, h3, h4, h5, h6, p, div, span, button, input, textarea, select, .stChatMessage {
        font-family: "Cairo", sans-serif !important;
    }   
    */
    ul{
    list-style-type: none; /* إزالة النقاط من القوائم */

    }

    /* ... بقية الـ CSS (الخلفية، الفقاعات، إلخ) ... */


    /* ... (كود CSS الخاص بك هنا) ... */
    /* الألوان من الثيم:
    primaryColor = #9cc6e4 (أزرق سماوي)
    backgroundColor = #082c58 (أزرق داكن)
    secondaryBackgroundColor = #183b65 (أزرق أفتح)
    textColor = #FFFFFF (أبيض)
    accentColor = #b9955c (ذهبي/بني)
    */

    /* إخفاء شعار Streamlit في التذييل والرأس */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ضبط حجم ومحاذاة العنوان الرئيسي */
    h1 {
        text-align: center;
        color: #9cc6e4; 
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }

    /* خلفية ومظهر شريط الدردشة */
    div.stChatInputContainer {
        background-color: #183b65; 
        border-top: 1px solid #082c58; 
        padding: 10px 0;
        position: fixed; 
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        padding-left: 20px;
        padding-right: 20px;
    }
    div.stChatInputContainer > div > div > div > div > div > div {
        margin: auto;
        max-width: 700px; 
    }

    /* Scrollbar للمحادثة */
    .main .block-container {
        padding-bottom: 100px; 
    }

    /* فقاعات الدردشة */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .stChatMessage[data-testid="stChatMessage"][data-state="user"] {
        background-color: #9cc6e4; 
        color: #082c58; 
        border-bottom-right-radius: 5px;
        margin-left: 20%; 
        text-align: right;
    }
    .stChatMessage[data-testid="stChatMessage"][data-state="assistant"] {
        background-color: #183b65; 
        color: #FFFFFF; 
        border-bottom-left-radius: 5px;
        margin-right: 20%; 
        text-align: left;
    }
    
    /* ضبط مظهر المؤشر والتحميل */
    .stSpinner > div > div {
        color: #b9955c !important; 
    }

    /* تحسين مظهر الأزرار */
    .stButton > button {
        background-color: #9cc6e4;
        color: #082c58;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s, transform 0.2s;
        width: 100%; 
        margin-bottom: 10px; 
    }
    .stButton > button:hover {
        background-color: #b9955c; 
        color: #FFFFFF; 
        transform: translateY(-2px);
    }

    /* تحسين مظهر الـ info, success, error messages */
    .stAlert {
        border-radius: 10px;
    }
    .stAlert > div > div {
        background-color: #183b65 !important; 
        color: #FFFFFF !important; 
    }
    .stAlert [data-testid="stIcon"] {
        color: #9cc6e4 !important; 
    }

    /* رسالة الترحيب في البداية */
    .welcome-message {
        background-color: #082c58; 
        color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 20px;
        border: 1px solid #183b65;
    }
    
    /* تنسيق الشريط الجانبي نفسه */
    section[data-testid="stSidebar"] {
        background-color: #183b65; 
        color: #FFFFFF; 
        padding-top: 20px; 
        padding-left: 10px;
        padding-right: 10px;
        border-right: 1px solid #082c58; 
        direction: rtl; /* محاذاة النصوص إلى اليمين */
    }

    /* تنسيق محتوى الشريط الجانبي (مثل العنوان الفرعي) */
    .st-emotion-cache-h6n3qj { 
        color: #9cc6e4 !important; 
        margin-bottom: 15px; 
    }

    /* تنسيق قسم "حول البوت" المحدد في الشريط الجانبي */
    .st-emotion-cache-16nr0lz { 
        background-color: rgba(0, 0, 0, 0.2); 
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #082c58; 
    }
    /* تنسيق الأزرار داخل الـ expander في الشريط الجانبي */
    .st-emotion-cache-1cpx6h0 { 
        color: #FFFFFF;
        background-color: #082c58; 
        border-radius: 8px;
        padding: 10px;
    }
    .st-emotion-cache-1cpx6h0:hover {
        background-color: #b9955c !important;
        color: #FFFFFF !important;
    }
    .st-emotion-cache-1cpx6h0 [data-testid="stExpanderChevron"] {
        color: #9cc6e4 !important; 
    }


        /* إضافة الخط الجديد */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@200..1000&display=swap');
    body, html, .st-emotion-cache-ch5fby, h1, h2, h3, h4, h5, h6, p, div, span, button, input, textarea, select, .stChatMessage {
        font-family: "Cairo", sans-serif !important;
    }

    /* تطبيق تدرج الخلفية الجديد */
    body {
        background: radial-gradient(circle at 20% 100%,
            rgba(184, 184, 184, 0.1) 0%,
            rgba(184, 184, 184, 0.1) 33%,
            rgba(96, 96, 96, 0.1) 33%,
            rgba(96, 96, 96, 0.1) 66%,
            rgba(7, 7, 7, 0.1) 66%,
            rgba(7, 7, 7, 0.1) 99%),
            linear-gradient(40deg, #040a22, #002954, #061861, #0376c1);
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #FFFFFF; /* نص أبيض */
    }
    .main {
        background: none !important; 
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --- اسم ملف سجل المحادثات الدائم ---
CHAT_HISTORY_FILE = "chat_history.json"

# --- دالة لتحميل سجل المحادثات من الملف ---
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError: 
                return []
    return []

# --- دالة لحفظ سجل المحادثات في الملف ---
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# --- تهيئة سجل المحادثات في Streamlit's session_state ---
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history() 
    if not st.session_state.messages: 
        st.session_state.messages.append(
            {"role": "assistant", "content": "أهلاً بك! أنا شات بوت جامعة الشام. كيف يمكنني مساعدتك اليوم؟", "timestamp": time.time()} 
        )


# --- الشريط الجانبي (Sidebar) ---
with st.sidebar:
    with st.container(): 
        st.image("images/logo.png", use_container_width=True) 
        st.markdown("<h3 style='color: #9cc6e4;'>حول البوت ✨</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            **مساعد جامعة الشام الذكي** 🤖
            \n
            يهدف هذا البوت إلى توفير معلومات سريعة وموثوقة حول:
            * 🏛️ الكليات والأقسام
            * 📝 شروط القبول والتسجيل
            * 📅 الفعاليات والأخبار
            * 📞 معلومات الاتصال
            \n
            **كيف يعمل؟**
            يعتمد البوت على قاعدة بيانات مُجهزة بعناية من **الأسئلة الشائعة (FAQ)** للجامعة، بالإضافة إلى **الذكاء الاصطناعي التوليدي من Google Gemini** للإجابة على الأسئلة الأخرى.
            """
        )
        st.markdown("<div style='color: #b9955c; font-size: 0.9em;'>تذكر: المعلومات دقيقة بناءً على المصدر.</div>", unsafe_allow_html=True)
        st.markdown("---") 
    st.markdown("<h3 style='color: #9cc6e4;'>تم بناء هذا الشات من قبل الطالبين</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #ffb563;text-shadow: 0 0 5px red;'>Hamid Al-murai & Ahmed Al-Zain</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #9cc6e4;text-shadow: 0 0 3px #a2d2ff;'>بإشراف الدكتورة هنادي العبد الله</h3>", unsafe_allow_html=True)
    st.markdown("---")



    st.subheader("خيارات المحادثة")
    
    if st.button("💬 بدء محادثة جديدة", help="ابدأ محادثة جديدة بمسح السجل الحالي.", key="new_chat_button"):
        st.session_state.messages = [] 
        st.session_state.messages.append(
            {"role": "assistant", "content": "أهلاً بك! أنا شات بوت جامعة الشام. كيف يمكنني مساعدتك اليوم؟", "timestamp": time.time()} 
        )
        st.rerun() 

    if st.button("🗑️ مسح جميع المحادثات (دائم)", help="مسح سجل المحادثات بالكامل من الذاكرة والملف بشكل دائم.", key="clear_all_chats_button"):
        st.session_state.messages = [] 
        save_chat_history([]) 
        st.rerun() 

    st.markdown("---")
    st.subheader("سجل المحادثات السابق")

    if "show_history_expanded" not in st.session_state:
        st.session_state.show_history_expanded = False

    if st.button("👁️ عرض سجل المحادثات", help="عرض جميع المحادثات المحفوظة.", key="show_history_button"):
        st.session_state.show_history_expanded = not st.session_state.show_history_expanded

    if st.session_state.show_history_expanded:
        with st.expander("السجل الكامل للمحادثات", expanded=True):
            full_history_data = load_chat_history()
            if full_history_data:
                for i, msg in enumerate(full_history_data):
                    if "role" in msg and "content" in msg and "timestamp" in msg:
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(msg["timestamp"]))
                        with st.chat_message(msg["role"]):
                            st.markdown(f"**[{timestamp}]**: {msg['content']}")
                    else:
                        st.warning(f"⚠️ رسالة ذات تنسيق خاطئ في السجل (اندكس {i}): {msg}")
            else:
                st.info("لا يوجد سجل محادثات لعرضه بعد.")


# --- العنوان الرئيسي والقسم الترحيبي (تم نقلهما إلى الأعلى) ---
st.title("🎓 شات بوت جامعة الشام")
st.markdown("---") 

if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
    st.markdown(
        """
        <div class="welcome-message">
        👋 **مرحباً بك في مساعد جامعة الشام الذكي!**<br>
        أنا هنا لأجيب على جميع استفساراتك حول الكليات، الأقسام، شروط القبول، والمزيد.<br>
        فقط اطرح سؤالك وسأبذل قصارى جهدي لمساعدتك بناءً على المعلومات المتوفرة لدي.
        </div>
        """,
        unsafe_allow_html=True
    )


# --- 1. تهيئة الشات بوت والأدوات ---
qa_vector_db_path = "faiss_university_qa_db" 
try:
    # model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    model_name = "asafaya/bert-base-arabic" # نموذج عربي من Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("❌ خطأ: متغير البيئة 'GOOGLE_API_KEY' غير موجود في ملف .env أو البيئة. يرجى تعيينه لتشغيل الذكاء الاصطناعي التوليدي.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # استخدام نموذج Google Gemini
    st.success("✔ تم تهيئة نموذج Google Gemini LLM (للذكاء الاصطناعي التوليدي).")

    faq_vector_db = FAISS.load_local(qa_vector_db_path, embeddings, allow_dangerous_deserialization=True)
    st.success("✔ تم تحميل قاعدة بيانات الأسئلة والأجوبة (FAQ) بنجاح.")

    # --- تهيئة أداة البحث على الويب (Google Serper) ---
    if "SERPER_API_KEY" not in st.secrets:
        st.warning("⚠️ تحذير: متغير البيئة 'SERPER_API_KEY' غير موجود. لن يتم تفعيل أداة البحث على الويب.")
        serper_search = None
    else:
        serper_search = GoogleSerperAPIWrapper(gl="sa", hl="ar", k=3) # البحث باللغة العربية، أول 3 نتائج
        st.success("✔ تم تهيئة أداة البحث على الويب (Google Serper).")

except Exception as e:
    st.error(f"❌ خطأ فادح: لم يتمكن من تحميل قواعد بيانات المعرفة أو تهيئة LLM/الأدوات. يرجى التأكد من أن المسار صحيح وأن المفاتيح API صحيحة. ({e})")
    st.stop() 


# --- 2. دالة البحث عن المعلومات (Multilayered Search) ---
def get_bot_response(user_question):
    # المرحلة 1: البحث في قاعدة بيانات الأسئلة والأجوبة (FAQ) الخاصة بنا
    docs_with_scores = faq_vector_db.similarity_search_with_score(user_question, k=1)
    
    best_faq_doc = None
    best_faq_score = float('inf') 
    
    if docs_with_scores:
        best_faq_doc, best_faq_score = docs_with_scores[0]
        st.sidebar.markdown(f"**<span style='color:#9cc6e4;'>أفضل مسافة في الـ FAQ:</span> {best_faq_score:.2f}**", unsafe_allow_html=True) 
        st.sidebar.write(f"DEBUG: أفضل تطابق FAQ (سؤال): '{best_faq_doc.page_content}'") 

    if best_faq_doc and best_faq_score <= FAQ_DISTANCE_THRESHOLD: 
        if "answer" in best_faq_doc.metadata:
            st.sidebar.write(f"DEBUG: تم الإجابة من FAQ مباشرة (المسافة: {best_faq_score:.2f} <= {FAQ_DISTANCE_THRESHOLD}).") 
            return best_faq_doc.metadata["answer"], "faq" 
        else:
            st.sidebar.warning("⚠️ تم العثور على سؤال FAQ، لكنه لا يحتوي على إجابة. سيتم اللجوء للبحث على الويب أو LLM.")
    
    # المرحلة 2: البحث في المواقع التعريفية للجامعة (عبر Google Serper)
    # نستخدم أداة بحث الويب للبحث عن معلومات تتعلق بالجامعة
    # يمكننا تضييق نطاق البحث على نطاق الجامعة (site:shamuniversity.com)
    # أو تضمين روابط الفيسبوك والتليجرام (ولكن البحث داخلها ليس مباشرا)
    
    # نختبر ما إذا كانت الأداة متاحة
    if serper_search:
        # بناء استعلام البحث: نركز على نطاق الجامعة أو الأسئلة العامة حول الجامعة
        search_query_base = f"جامعة الشام {user_question}"
        # يمكن تحديد نطاقات محددة للبحث:
        # search_query_site_specific = f"{user_question} site:shamuniversity.com OR site:t.me/InterShamUniv OR site:www.facebook.com/SHAM.UNIV"
        # لكن البحث داخل التليجرام والفيسبوك مباشرة صعب عبر Serper

        st.sidebar.write(f"DEBUG: لا يوجد تطابق FAQ بثقة عالية. جاري البحث على الويب عن '{search_query_base}'...")
        try:
            # استخدام أداة البحث
            search_results = serper_search.run(search_query_base)
            if search_results and "answer_box" in search_results and search_results["answer_box"]:
                answer_from_serper = search_results["answer_box"].get("snippet", search_results["answer_box"].get("answer"))
                if answer_from_serper:
                    st.sidebar.write(f"DEBUG: تم العثور على إجابة من Serper (صندوق الإجابة).")
                    return answer_from_serper, "web_search_answer_box"
            
            # إذا لم يكن هناك صندوق إجابة مباشر، حاول تلخيص النتائج العادية
            if search_results and "snippets" in search_results:
                relevant_snippets = []
                for snippet in search_results["snippets"]:
                    # فلترة النتائج لتكون ذات صلة بالجامعة قدر الإمكان
                    if "shamuniversity.com" in snippet.get("link", "") or "SHAM.UNIV" in snippet.get("link", ""):
                        relevant_snippets.append(snippet.get("snippet"))
                
                if relevant_snippets:
                    combined_snippets = "\n".join(relevant_snippets[:3]) # خذ أول 3 مقتطفات ذات صلة
                    st.sidebar.write(f"DEBUG: جاري تلخيص نتائج البحث من Serper.")
                    
                    # استخدام LLM لتلخيص النتائج
                    summary_prompt = ChatPromptTemplate.from_template(
                        """
                        أنت مساعد ذكي. لخص المعلومات التالية للإجابة على السؤال التالي.
                        إذا كانت المعلومات غير كافية، اذكر ذلك بوضوح.
                        السؤال: {question}
                        المعلومات المسترجعة:
                        {snippets}
                        
                        الإجابة الملخصة:
                        """
                    )
                    summary_chain = summary_prompt | llm | StrOutputParser()
                    summarized_answer = summary_chain.invoke({"question": user_question, "snippets": combined_snippets})
                    if summarized_answer:
                        return summarized_answer, "web_search_summary"
            
            st.sidebar.write("DEBUG: Serper لم يجد إجابة مباشرة أو مقتطفات كافية. اللجوء إلى LLM.")

        except Exception as serper_e:
            st.sidebar.error(f"❌ حدث خطأ أثناء البحث على الويب: {serper_e}")
            st.sidebar.write("DEBUG: حدث خطأ في Serper. اللجوء إلى LLM.")


    # المرحلة 3: اللجوء إلى LLM (الذكاء الاصطناعي العام) كخيار أخير
    general_llm_prompt = ChatPromptTemplate.from_template(
        """
        أنت مساعد ذكي. أجب على السؤال بأسلوب مهذب وواضح.
        إذا كان السؤال يتعلق بمعلومات محددة جداً لا تعرفها ولم تجدها في مصادر البحث، اذكر بوضوح أنك لا تعرف.
        لا تحاول اختلاق إجابات.

        سؤال المستخدم: {question}

        الإجابة:
        """
    )
    general_response_chain = general_llm_prompt | llm | StrOutputParser()
    try:
        llm_answer = general_response_chain.invoke({"question": user_question})
        return llm_answer, "llm" 
    except Exception as llm_e:
        st.error(f"❌ حدث خطأ أثناء الاتصال بالذكاء الاصطناعي التوليدي: {llm_e}")
        return "عذراً، حدث خطأ أثناء محاولة الإجابة من الذكاء الاصطناعي. يرجى المحاولة لاحقاً.", "error" 

# --- 3. عرض سجل المحادثات السابق (كما هو) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. صندوق إدخال السؤال الجديد (Chat Input) (كما هو) ---
if user_question := st.chat_input("اطرح سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": user_question, "timestamp": time.time()})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث عن الإجابة... من فضلك انتظر ⏳"): 
            try:
                answer_content, answer_source = get_bot_response(user_question) 

                source_indicator_message = ""
                if answer_source == "faq":
                    source_indicator_message = "<p style='font-size: 0.8em; color: #b9955c; margin-bottom: 5px;'><i>(الإجابة من قاعدة بيانات الأسئلة الشائعة 📚)</i></p>"
                elif answer_source == "web_search_answer_box":
                    source_indicator_message = "<p style='font-size: 0.8em; color: #66CCFF; margin-bottom: 5px;'><i>(الإجابة من بحث الويب 🌐)</i></p>"
                elif answer_source == "web_search_summary":
                    source_indicator_message = "<p style='font-size: 0.8em; color: #66CCFF; margin-bottom: 5px;'><i>(الإجابة ملخصة من نتائج بحث الويب 🌐)</i></p>"
                elif answer_source == "llm":
                    source_indicator_message = "<p style='font-size: 0.8em; color: #9cc6e4; margin-bottom: 5px;'><i>(الذكاء الاصطناعي التوليدي قام بالإجابة ✨)</i></p>"
                elif answer_source == "error":
                    source_indicator_message = "<p style='font-size: 0.8em; color: red; margin-bottom: 5px;'><i>(حدث خطأ أثناء الحصول على الإجابة 🚫)</i></p>"

                if source_indicator_message:
                    st.markdown(source_indicator_message, unsafe_allow_html=True)
                
                st.markdown(answer_content) 

                st.session_state.messages.append({"role": "assistant", "content": answer_content, "timestamp": time.time()})
                
                full_history = load_chat_history()
                full_history.append({"timestamp": time.time(), "role": "user", "content": user_question})
                full_history.append({"timestamp": time.time(), "role": "assistant", "content": answer_content})
                save_chat_history(full_history)

            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة سؤالك: {e}")
                st.error("الرجاء التأكد من أن مفتاح Google API الخاص بك صحيح ويعمل، وأن قاعدة بيانات الأسئلة والأجوبة موجودة وصحيحة، وأن مفتاح Serper API صحيح إذا كنت تستخدمه.")


# --- قسم تذييل الصفحة (Footer) ---
st.markdown("---")
st.markdown(
    "**ملاحظة هامة:** هذا الشات بوت يجيب بناءً على معلومات تم جمعها وتجهيزها مسبقاً من موقع الجامعة. "
    "للحصول على أحدث المعلومات أو في حالات الطوارئ، يرجى زيارة الموقع الرسمي أو التواصل المباشر مع الجامعة."
)
# إضافة روابط سريعة للمواقع التعريفية مع الأيقونات
st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 1.1em;'>
    روابط مهمة: 
    <a href="https://shamuniversity.com/" target="_blank" style="color:#9cc6e4; margin: 0 5px;" title="الموقع الرسمي"><i class="fas fa-globe"></i></a> | 
    <a href="https://t.me/InterShamUniv" target="_blank" style="color:#9cc6e4; margin: 0 5px;" title="قناة التليجرام"><i class="fab fa-telegram-plane"></i></a> | 
    <a href="https://www.facebook.com/SHAM.UNIV/?locale=ar_AR" target="_blank" style="color:#9cc6e4; margin: 0 5px;" title="صفحة الفيسبوك"><i class="fab fa-facebook"></i></a> |
    <a href="https://www.instagram.com/unvalsham/" target="_blank" style="color:#9cc6e4; margin: 0 5px;" title="صفحة الإنستغرام"><i class="fab fa-instagram"></i></a> |
    <a href="https://maps.app.goo.gl/d3FJsTHkAJmyGosH6" target="_blank" style="color:#9cc6e4; margin: 0 5px;" title="الموقع الجغرافي للجامعة"><i class="fas fa-map-marker-alt"></i></a>
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='text-align: center; color: gray;'>© 2025 شات بوت جامعة الشام</div>", unsafe_allow_html=True)
