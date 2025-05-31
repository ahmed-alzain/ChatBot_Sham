import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. تحميل قاعدة بيانات المتجهات المحفوظة (الجديدة للـ FAQ)
vector_db_path = "faiss_university_qa_db" # المسار الجديد لقاعدة بيانات الـ QA
try:
    # model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    model_name = "asafaya/bert-base-arabic"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    # سنسترجع مستنداً واحداً فقط (أفضل سؤال مطابق)
    retriever = vector_db.as_retriever(search_kwargs={"k": 1}) 
    print(f"تم تحميل قاعدة بيانات المتجهات من: {vector_db_path}")

except Exception as e:
    print(f"خطأ: لم يتمكن من تحميل قاعدة بيانات المتجهات. يرجى التأكد من أن المسار صحيح وأن الملفات موجودة. {e}")
    print("قد تحتاج إلى إعادة تشغيل 'build_vector_db.py' مع ملف الـ FAQ الجديد.")
    exit()

# 2. بناء سلسلة الشات بوت القائمة على الأسئلة والأجوبة (Retrieval only)
def get_answer_from_retrieved_docs(docs):
    if not docs:
        return "عذراً، لم أجد إجابة محددة لهذا السؤال في المعلومات المتوفرة."

    first_doc = docs[0]
    if "answer" in first_doc.metadata and first_doc.metadata["answer"]:
        return first_doc.metadata["answer"]
    else:
        return "عذراً، لم أجد إجابة محددة لهذا السؤال في المعلومات المتوفرة."

qa_retrieval_chain = (
    retriever 
    | get_answer_from_retrieved_docs
)

print("تم بناء سلسلة الشات بوت القائمة على الأسئلة والأجوبة (FAQ).")

# 3. حلقة التفاعل مع الشات بوت
print("\n--- مرحباً بك في شات بوت جامعة الشام! ---")
print("يمكنك طرح أي سؤال عن الجامعة، أو اكتب 'exit' للخروج.")

while True:
    user_question = input("\nسؤالك: ")
    if user_question.lower() == 'exit':
        print("شكراً لاستخدامك شات بوت جامعة الشام. إلى اللقاء!")
        break

    try:
        response = qa_retrieval_chain.invoke(user_question)
        print(f"الروبوت: {response}")
    except Exception as e:
        print(f"حدث خطأ أثناء معالجة سؤالك: {e}")
        print("الرجاء التأكد من أن قاعدة بيانات الأسئلة والأجوبة موجودة وصحيحة.")