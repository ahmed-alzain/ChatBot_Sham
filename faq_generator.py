import os
from turtle import st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import json
from tqdm import tqdm # استيراد tqdm

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

class FAQGenerator:
    def __init__(self, input_text_files, output_faq_file="generated_university_faq.txt"):
        if not isinstance(input_text_files, list) or not all(isinstance(f, str) for f in input_text_files):
            raise ValueError("input_text_files يجب أن تكون قائمة من مسارات الملفات النصية.")
            
        self.input_text_files = input_text_files
        self.output_faq_file = output_faq_file
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if "GOOGLE_API_KEY" not in st.secrets:
            raise ValueError("خطأ: متغير البيئة 'GOOGLE_API_KEY' غير موجود. يرجى تعيينه.")
        print("✔ تم تهيئة نموذج Google Gemini LLM (للذكاء الاصطناعي التوليدي).")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) 

    def _load_text_data(self):
        all_paragraphs = []
        for file_path in self.input_text_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    paragraphs_from_file = [line.strip() for line in f if line.strip()]
                all_paragraphs.extend(paragraphs_from_file)
                print(f"تم قراءة {len(paragraphs_from_file)} فقرة من الملف '{file_path}'.")
            except FileNotFoundError:
                print(f"خطأ: ملف الإدخال '{file_path}' غير موجود. يرجى التأكد من تشغيل سكربتات الزحف والتنظيف أولاً.")
            except Exception as e:
                print(f"حدث خطأ أثناء قراءة ملف الإدخال '{file_path}': {e}")
        
        unique_paragraphs = list(set(all_paragraphs))
        print(f"إجمالي الفقرات الفريدة المدمجة من جميع الملفات: {len(unique_paragraphs)}.")
        return unique_paragraphs

    def _generate_qa_for_chunk(self, chunk):
        prompt_template = ChatPromptTemplate.from_template(
            """
            أنت مساعد متخصص في توليد الأسئلة والأجوبة من النصوص.
            المهمة: استخرج من النص التالي 2 إلى 5 أسئلة وأجوبة محددة وموجزة.
            يجب أن تكون الإجابة مباشرة من النص.
            كل سؤال وجواب يجب أن يكونا على سطرين منفصلين.
            استخدم التنسيق التالي بدقة:
            س: [السؤال هنا]
            ج: [الإجابة هنا]
            ---
            س: [السؤال التالي هنا]
            ج: [الإجابة التالية هنا]
            ---
            ...
            
            النص:
            {text_chunk}

            الأسئلة والأجوبة:
            """
        )

        qa_generation_chain = prompt_template | self.llm | StrOutputParser()

        try:
            response_text = qa_generation_chain.invoke({"text_chunk": chunk})
            generated_pairs = []
            
            entries = response_text.strip().split("---")
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                
                question_match = re.search(r"س:\s*(.*?)\n", entry, re.DOTALL)
                answer_match = re.search(r"ج:\s*(.*)", entry, re.DOTALL) 

                if question_match and answer_match:
                    question = question_match.group(1).strip()
                    answer = answer_match.group(1).strip()
                    if question and answer: 
                        generated_pairs.append({"question": question, "answer": answer})
                else:
                    # لا تطبع تحذير لكل قطعة، يمكن أن تكون مزعجة
                    # print(f"⚠️ تحذير: لم يتم تحليل زوج سؤال-جواب بشكل صحيح من الجزء: \n{entry[:150]}...") 
                    pass # تجاهل التحذيرات هنا لتحسين نظافة المخرجات أثناء التشغيل
            return generated_pairs
        except Exception as e:
            print(f"❌ حدث خطأ أثناء توليد QA لجزء النص: {e}")
            return []

    def generate_faqs(self):
        paragraphs = self._load_text_data()
        if not paragraphs:
            print("لا توجد فقرات لمعالجتها. يرجى التأكد من أن ملفات الإدخال تحتوي على بيانات.")
            return

        all_generated_faqs = []
        
        current_chunk = ""
        chunk_size_limit = 1500 
        
        print("\nبدء توليد الأسئلة والأجوبة من الفقرات المدمجة...")
        
        # شريط تقدم لعملية توليد الأسئلة والأجوبة
        # نقسم الفقرات إلى قطع ثم نعالجها
        # التقدير التقريبي لعدد طلبات الـ LLM
        num_chunks = (len("".join(paragraphs)) + chunk_size_limit - 1) // chunk_size_limit
        if num_chunks == 0: num_chunks = 1 # لضمان شريط التقدم حتى لو كانت الفقرات قليلة

        with tqdm(total=num_chunks, unit="chunk", desc="توليد الأسئلة والأجوبة") as pbar:
            for i, paragraph in enumerate(paragraphs):
                current_chunk += paragraph + "\n" 
                if len(current_chunk) >= chunk_size_limit or i == len(paragraphs) - 1:
                    qas = self._generate_qa_for_chunk(current_chunk.strip())
                    all_generated_faqs.extend(qas)
                    current_chunk = "" 
                    pbar.update(1) # تحديث شريط التقدم بعد معالجة كل قطعة
                    pbar.set_postfix_str(f"FAQ مجمعة: {len(all_generated_faqs)}") # عرض عدد الـ FAQ المجمعة

        print("\nعملية توليد FAQ انتهت!")
        print(f"تم توليد {len(all_generated_faqs)} زوج سؤال-جواب (قد تحتوي على تكرارات قبل الفلترة النهائية).")
        
    def _save_faqs_to_file(self, faqs):
        unique_faq_contents = set()
        
        with open(self.output_faq_file, "w", encoding="utf-8") as f:
            for faq_pair in faqs:
                question = faq_pair.get("question", "").strip()
                answer = faq_pair.get("answer", "").strip()
                
                if question and answer:
                    faq_entry = f"س: {question}\nج: {answer}\n---"
                    if faq_entry not in unique_faq_contents:
                        f.write(faq_entry + "\n") 
                        unique_faq_contents.add(faq_entry)
        print(f"تم حفظ {len(unique_faq_contents)} زوج سؤال-جواب فريد في الملف '{self.output_faq_file}'.")


# --- كيفية الاستخدام (كما هو) ---
if __name__ == "__main__":
    input_files_for_faq_generation = [
        "all_cleaned_university_paragraphs.txt" # هذا هو ملف الإدخال الوحيد الآن
    ]
    output_faq_filename = "generated_university_faq.txt" 

    try:
        generator = FAQGenerator(
            input_text_files=input_files_for_faq_generation, 
            output_faq_file=output_faq_filename
        )
        generator.generate_faqs()

        print("\n--- الخطوات التالية المقترحة: ---")
        print(f"1. راجع الملف '{output_faq_filename}' للتأكد من جودة الأسئلة والأجوبة المولدة.")
        print(f"2. إذا كنت راضياً، يمكنك استخدام هذا الملف بدلاً من 'university_faq_qa.txt' الأصلي (أو دمجهما يدوياً).")
        print(f"3. قم بتشغيل 'python build_vector_db.py' لإنشاء قاعدة بيانات FAISS جديدة بناءً على ملف الـ FAQ المحدث.")
        print("   (تأكد من أن build_vector_db.py يستخدم المسار الصحيح لملف الـ FAQ الجديد الخاص بك،")
        print("   أو قم بتسمية generated_university_faq.txt باسم university_faq_qa.txt).")
        print("4. يمكنك الآن تشغيل 'app.py' وسيقوم الشات بوت باستخدام قاعدة بيانات FAQ المحدثة تلقائياً.")

    except ValueError as ve:
        print(f"خطأ في التهيئة: {ve}")
    except Exception as e:
        print(f"حدث خطأ غير متوقع أثناء تشغيل المولد: {e}")