import re
import os
from tqdm import tqdm # لاستخدام شريط التقدم

def clean_text_data_single_paragraph(text_input):
    """
    تقوم بتنظيف فقرة نصية واحدة.
    تم فصل هذه الدالة لتكون مستقلة ويمكن استدعاؤها بسهولة.
    """
    unwanted_phrases = [
        "نهتم دوما بالاستماع إلى مقترحاتكم وآرائكم.",
        "Copyright ©جميع الحقوق محفوظة لجامعة الشام",
        "جامعة الشام",
        "Sham university",
        "المزيد", 
        "Copyright ©جميع الحقوق محفوظة لمركز شام للدراسات والبحث العلمي",
        "Sham university"
    ]

    text = text_input.strip()
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
    
    # الاحتفاظ بالحروف العربية، الإنجليزية، الأرقام، المسافات، النقطة، والفاصلة
    text = re.sub(r'[^\u0600-\u06FF\sA-Za-z0-9\.\,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower() # توحيد الحروف (خاصة للإنجليزية)

    # إزالة الفقرات الفارغة أو القصيرة جداً
    if text and len(text) > 10: 
        return text
    return None

# --- الجزء الخاص بتشغيل السكربت مباشرة ---
if __name__ == "__main__":
    # تعريف ملفات الإدخال الخام التي تريد تنظيفها ودمجها
    input_files_to_clean = [
        "all_university_paragraphs.txt",       # الناتج من scrape_sham_university.py
        "university_texts_with_ocr.txt"        # الناتج من scrape_with_ocr.py
    ]
    
    # الملف النهائي الذي سيحتوي على جميع الفقرات النظيفة والفريدة والمدمجة
    final_output_cleaned_file = "all_cleaned_university_paragraphs.txt"

    all_raw_paragraphs_to_clean = []

    print("بدء عملية قراءة الفقرات من ملفات الإدخال...")

    # قراءة الفقرات من جميع ملفات الإدخال
    for file_path in input_files_to_clean:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    paragraphs_from_file = [line.strip() for line in f if line.strip()]
                    all_raw_paragraphs_to_clean.extend(paragraphs_from_file)
                print(f"تم قراءة {len(paragraphs_from_file)} فقرة من الملف '{file_path}'.")
            except Exception as e:
                print(f"خطأ: حدث مشكلة في قراءة الملف '{file_path}': {e}")
        else:
            print(f"⚠️ تحذير: ملف '{file_path}' غير موجود. سيتم تخطيه في عملية التنظيف.")

    if not all_raw_paragraphs_to_clean:
        print("لا توجد بيانات لمعالجتها من أي من ملفات الإدخال المحددة. تأكد من تشغيل سكربتات الزحف أولاً.")
        exit()

    print(f"\nبدء عملية تنظيف ودمج {len(all_raw_paragraphs_to_clean)} فقرة خام...")
    cleaned_unique_paragraphs_set = set() # استخدام مجموعة لضمان التفرد بعد التنظيف

    # تطبيق التنظيف على كل فقرة وعرض شريط تقدم
    with tqdm(total=len(all_raw_paragraphs_to_clean), unit="فقرة", desc="تنظيف ودمج الفقرات") as pbar:
        for paragraph in all_raw_paragraphs_to_clean:
            cleaned_text = clean_text_data_single_paragraph(paragraph)
            if cleaned_text:
                cleaned_unique_paragraphs_set.add(cleaned_text)
            pbar.update(1) # تحديث شريط التقدم

    cleaned_paragraphs_list = list(cleaned_unique_paragraphs_set)
    print(f"\nتم تنظيف ودمج واستخراج {len(cleaned_paragraphs_list)} فقرة فريدة نهائية.")

    # حفظ الفقرات النظيفة المدمجة في ملف جديد
    with open(final_output_cleaned_file, "w", encoding="utf-8") as f:
        for p in cleaned_paragraphs_list:
            f.write(p + "\n")
    print(f"تم حفظ جميع الفقرات النظيفة والفريدة والمدمجة في الملف: {final_output_cleaned_file}")

    print("\n--- الخطوات التالية المقترحة: ---")
    print(f"1. الآن، ملف '{final_output_cleaned_file}' يحتوي على جميع النصوص النظيفة والفريدة من HTML و OCR.")
    print(f"2. قم بتعديل 'faq_generator.py' ليشير إلى '{final_output_cleaned_file}' كمدخل وحيد.")
    print("   (افتح faq_generator.py، ابحث عن input_files_for_faq_generation وعدّلها).")
    print("3. شغّل 'python faq_generator.py' لتوليد الأسئلة والأجوبة.")
    print("4. راجع ملف الـ FAQ المولّد (generated_university_faq.txt) يدويًا، ثم قم بإعادة تسميته إلى 'university_faq_qa.txt'.")
    print("5. شغّل 'python build_vector_db.py' لإعادة بناء قاعدة بيانات المتجهات.")
    print("6. شغّل 'streamlit run app.py' لتشغيل الشات بوت.")