import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib3
import pytesseract
from PIL import Image
import io
import os
import time
from tqdm import tqdm # لاستخدام شريط التقدم

# --- إعداد Tesseract OCR في بايثون (تأكد من تعديل هذا المسار) ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ALWAFER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe' 

# إخفاء تحذير InsecureRequestWarning عند تعطيل التحقق من SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# قائمة لتخزين جميع الفقرات النصية المستخلصة
all_extracted_ocr_texts = []
# مجموعة لتتبع الروابط التي تمت زيارتها لتجنب التكرار في الزحف
visited_urls_ocr = set() # اسم مختلف لتجنب التداخل مع سكربت آخر إذا تم تشغيلهما بالتوازي
# مجموعة لتتبع صور تم معالجتها بـ OCR لتجنب تكرار معالجة نفس الصورة
processed_image_urls = set()
# قائمة انتظار للزحف (URL, depth)
crawl_queue_ocr = []

def extract_text_from_image(image_url):
    """
    يقوم بتحميل صورة من عنوان URL ويستخلص النص منها باستخدام Tesseract OCR.
    """
    if image_url in processed_image_urls: # تجنب معالجة نفس الصورة
        return ""
    processed_image_urls.add(image_url)

    try:
        img_response = requests.get(image_url, stream=True, timeout=10, verify=False)
        img_response.raise_for_status()
        image = Image.open(io.BytesIO(img_response.content))
        
        # تحويل الصورة إلى وضع تدرج رمادي لتحسين OCR
        image = image.convert('L') 
        
        text = pytesseract.image_to_string(image, lang='ara+eng')
        
        return text.strip()
    except requests.exceptions.RequestException as e:
        # print(f"خطأ في تحميل الصورة {image_url}: {e}")
        return ""
    except pytesseract.TesseractNotFoundError:
        print("خطأ: Tesseract OCR غير موجود. تأكد من تثبيته وإضافته إلى PATH.")
        return ""
    except Exception as e:
        # print(f"حدث خطأ أثناء معالجة الصورة {image_url}: {e}")
        return ""

def scrape_single_page_with_ocr(url):
    """
    يقوم بجمع الفقرات النصية من HTML والنصوص من الصور من صفحة ويب واحدة.
    """
    if url in visited_urls_ocr:
        return [] # تم زيارة هذه الصفحة من قبل

    visited_urls_ocr.add(url)

    page_texts_raw = []
    new_links_to_crawl = []

    try:
        response = requests.get(url, timeout=15, verify=False) 
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # استخلاص الروابط لمتابعة الزحف
        for link_tag in soup.find_all('a', href=True):
            href = link_tag.get('href')
            if href and not href.startswith('#'):
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    new_links_to_crawl.append(full_url)

        # استخلاص الفقرات النصية العادية من HTML
        for paragraph_tag in soup.find_all('p'):
            text = paragraph_tag.get_text(strip=True)
            if text:
                page_texts_raw.append(text)

        # استخلاص النصوص من الصور باستخدام OCR
        for img_tag in soup.find_all('img'):
            img_src = img_tag.get('src')
            if img_src:
                full_img_url = urljoin(url, img_src)
                # فلترة الصور الصغيرة جدًا أو الأيقونات لتجنب OCR غير الضروري
                # يمكنك تعديل هذه الشروط بناءً على تحليل موقعك
                if img_tag.get('width') and int(img_tag['width']) < 50 and img_tag.get('height') and int(img_tag['height']) < 50:
                    continue # تخطي الصور الصغيرة جداً
                
                extracted_img_text = extract_text_from_image(full_img_url)
                if extracted_img_text:
                    page_texts_raw.append(extracted_img_text)
                    # print(f"  - تم استخلاص نص من صورة: {full_img_url[:60]}...") 

        return new_links_to_crawl, page_texts_raw # ارجاع الروابط والنصوص الخام

    except requests.exceptions.Timeout:
        # print(f"انتهت مهلة الاتصال عند {url}")
        pass
    except requests.exceptions.RequestException as e:
        # print(f"حدث خطأ أثناء الاتصال بالخادم عند {url}: {e}")
        pass
    except Exception as e:
        # print(f"حدث خطأ غير متوقع عند {url}: {e}")
        pass
    return [], []

def crawl_ocr_website(start_urls, max_depth=3, output_file="university_texts_with_ocr.txt"):
    global crawl_queue_ocr, visited_urls_ocr, all_extracted_ocr_texts, processed_image_urls # للوصول للمتغيرات العالمية

    # إعادة تهيئة في حال تم التشغيل أكثر من مرة
    crawl_queue_ocr = [(url, 0) for url in start_urls if url not in visited_urls_ocr]
    visited_urls_ocr.clear() 
    all_extracted_ocr_texts.clear()
    processed_image_urls.clear()

    initial_queue_size = len(crawl_queue_ocr)
    
    print("\nبدء عملية الزحف واستخلاص النصوص من HTML والصور (OCR)...")

    with tqdm(total=initial_queue_size, unit="صفحة", desc="الزحف ومعالجة الصور") as pbar:
        while crawl_queue_ocr:
            current_url, current_depth = crawl_queue_ocr.pop(0)

            if current_url in visited_urls_ocr:
                pbar.update(1)
                continue

            links_from_page, texts_from_page = scrape_single_page_with_ocr(current_url)
            all_extracted_ocr_texts.extend(texts_from_page)
            
            pbar.update(1)
            pbar.set_postfix_str(f"فقرات OCR مجمعة: {len(all_extracted_ocr_texts)}")


            if current_depth < max_depth:
                for link in links_from_page:
                    if link not in visited_urls_ocr and (link, current_depth + 1) not in crawl_queue_ocr:
                        crawl_queue_ocr.append((link, current_depth + 1))
                        pbar.total += 1
                        pbar.refresh()

    print("\nعملية الزحف واستخلاص النصوص من الصور انتهت!")
    print(f"تم جمع {len(all_extracted_ocr_texts)} فقرة نصية (من HTML و OCR).")

    # حفظ جميع الفقرات المستخلصة في ملف
    with open(output_file, "w", encoding="utf-8") as f:
        for text_item in all_extracted_ocr_texts:
            f.write(text_item + "\n")
    print(f"تم حفظ نصوص OCR الخام في الملف: {output_file}")


# --- الروابط الأولية للبدء بالزحف ---
initial_urls_for_ocr_crawl = [
    "https://shamuniversity.com",
    "https://shamuniversity.com/nav14", 
    "https://shamuniversity.com/nav21", 
    "https://shamuniversity.com/nav28", 
    "https://shamuniversity.com/nav29", 
    "https://shamuniversity.com/nav24", 
    "https://shamuniversity.com/nav25", 
    "https://shamuniversity.com/nav15", 
    "https://shamuniversity.com/nav36", 
    "https://shamuniversity.com/nav41", 
    "https://shamuniversity.com/nav67", 
    "https://shamuniversity.com/nav52",
    "https://shamuniversity.com/navnone", 
    "https://shamuniversity.com/nav64"
]

# --- كيفية الاستخدام (تشغيل السكربت) ---
if __name__ == "__main__":
    output_ocr_file = "university_texts_with_ocr.txt"
    crawl_ocr_website(initial_urls_for_ocr_crawl, max_depth=3, output_file=output_ocr_file)