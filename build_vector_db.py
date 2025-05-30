import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm # استيراد tqdm

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. تحميل أزواج الأسئلة والأجوبة من ملف الـ FAQ
faq_file_path = "university_faq_qa.txt" 
qa_documents = []
try:
    with open(faq_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        entries = content.split("---")
        for entry in entries:
            entry = entry.strip()
            if entry:
                lines = entry.split('\n')
                if len(lines) >= 2:
                    question = lines[0].strip().replace("س:", "").strip()
                    answer = lines[1].strip().replace("ج:", "").strip()
                    if question and answer:
                        qa_documents.append(Document(page_content=question, metadata={"answer": answer, "type": "qa_pair"}))
    print(f"تم قراءة {len(qa_documents)} زوج سؤال-جواب من الملف '{faq_file_path}'.")
except FileNotFoundError:
    print(f"خطأ: ملف '{faq_file_path}' غير موجود. يرجى إنشاءه أولاً (عبر faq_generator.py والمراجعة).")
    qa_documents = []

if not qa_documents:
    print("لا توجد بيانات أسئلة وأجوبة لمعالجتها. يرجى التأكد من أن ملف الـ FAQ غير فارغ.")
    exit()

# 2. تقسيم النصوص إلى أجزاء (Chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
)

chunks = text_splitter.split_documents(qa_documents)

print(f"تم تقسيم النصوص إلى {len(chunks)} جزءًا (chunk).")

# 3. إنشاء تضمينات (Embeddings) باستخدام Hugging Face Model
print("جاري إنشاء تضمينات (embeddings) للنصوص باستخدام نموذج Hugging Face.")
# model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
model_name = "asafaya/bert-base-arabic" # نموذج عربي مناسب
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. بناء قاعدة بيانات المتجهات (Vector Database) باستخدام FAISS
try:
    # سيتم إنشاء التضمينات لـ chunks هنا
    # نضيف شريط تقدم لعملية بناء قاعدة البيانات
    print("بدء بناء قاعدة بيانات المتجهات (FAISS). قد يستغرق هذا بعض الوقت...")
    
    # يجب أن تكون chunks قائمة من Document objects.
    # FAISS.from_documents() تقوم بتوليد التضمينات لكل مستند.
    # يمكننا لف هذه العملية بـ tqdm.
    # نظرًا لأن FAISS.from_documents تقوم بالعملية داخليًا، فإننا نستخدم tqdm لتتبع التقدم
    # أثناء إرسال المستندات لإنشاء التضمينات.

    # طريقة لتقدير التقدم: نكرر على الـ chunks ونضيفها تدريجياً لـ FAISS
    # بدلاً من FAISS.from_documents() المباشرة التي لا توفر Hook للتقدم.
    # أو يمكننا بناءها من المستند الأول ثم إضافة البقية.

    if not chunks:
        print("لا توجد أجزاء لإنشاء قاعدة البيانات منها.")
        vector_db = None
    else:
        # إنشاء قاعدة البيانات من الجزء الأول
        vector_db = FAISS.from_documents([chunks[0]], embeddings)
        
        # إضافة الأجزاء المتبقية مع شريط تقدم
        if len(chunks) > 1:
            # استخدام tqdm حول باقي الأجزاء
            for i in tqdm(range(1, len(chunks)), unit="chunk", desc="إضافة تضمينات إلى DB"):
                vector_db.add_documents([chunks[i]])

    if vector_db:
        print("\nتم بناء قاعدة بيانات المتجهات بنجاح!")

        # 5. حفظ قاعدة بيانات المتجهات
        vector_db_path_qa = "faiss_university_qa_db" 
        vector_db.save_local(vector_db_path_qa)
        print(f"تم حفظ قاعدة بيانات المتجهات محلياً في: {vector_db_path_qa}")

        print("\nأصبحت قاعدة بيانات المتجهات للأسئلة والأجوبة جاهزة للاستعلامات!")
    else:
        print("لم يتم بناء قاعدة بيانات المتجهات بسبب عدم وجود أجزاء نصية.")

except Exception as e:
    print(f"حدث خطأ أثناء بناء قاعدة بيانات المتجهات: {e}")
    print("الرجاء التأكد من أن نموذج التضمين يمكن تحميله ويعمل بشكل صحيح.")