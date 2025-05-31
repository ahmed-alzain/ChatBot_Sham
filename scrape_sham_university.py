import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib3
import time
from tqdm import tqdm # لاستخدام شريط التقدم

# إخفاء تحذير InsecureRequestWarning عند تعطيل التحقق من SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# قائمة لتخزين جميع الفقرات النصية المستخلصة
all_extracted_paragraphs = []
# مجموعة لتتبع الروابط التي تمت زيارتها لتجنب التكرار في الزحف
visited_urls = set()
# قائمة انتظار للزحف (URL, depth)
crawl_queue = []

def scrape_single_html_page(url):
    """
    يقوم بجمع الروابط والفقرات النصية من صفحة ويب واحدة.
    """
    if url in visited_urls:
        return [], [] # تم زيارة هذه الصفحة من قبل

    visited_urls.add(url) 

    page_links = []
    page_paragraphs = []
    
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # استخلاص الروابط
        for link_tag in soup.find_all('a', href=True):
            href = link_tag.get('href')
            if href and not href.startswith('#'):
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    page_links.append(full_url)

        # استخلاص الفقرات النصية
        for paragraph_tag in soup.find_all('p'):
            text = paragraph_tag.get_text(strip=True)
            if text:
                page_paragraphs.append(text)

        return page_links, page_paragraphs

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

def crawl_html_website(start_urls, max_depth=3, output_file="all_university_paragraphs.txt"):
    global crawl_queue, visited_urls, all_extracted_paragraphs # للوصول للمتغيرات العالمية

    # إعادة تهيئة في حال تم التشغيل أكثر من مرة
    crawl_queue = [(url, 0) for url in start_urls if url not in visited_urls]
    visited_urls.clear() # مسح الروابط التي تمت زيارتها لكل عملية زحف
    all_extracted_paragraphs.clear() # مسح الفقرات المستخرجة لكل عملية زحف

    initial_queue_size = len(crawl_queue)
    
    print("\nبدء عملية الزحف لـ HTML فقط...")

    with tqdm(total=initial_queue_size, unit="صفحة", desc="الزحف على صفحات HTML") as pbar:
        while crawl_queue:
            current_url, current_depth = crawl_queue.pop(0)

            if current_url in visited_urls:
                pbar.update(1) # تحديث الشريط حتى لو تم تخطي الصفحة
                continue

            links_from_page, paragraphs_from_page = scrape_single_html_page(current_url)
            all_extracted_paragraphs.extend(paragraphs_from_page)
            
            pbar.update(1) # تحديث شريط التقدم بعد معالجة الصفحة
            pbar.set_postfix_str(f"فقرات HTML مجمعة: {len(all_extracted_paragraphs)}")

            if current_depth < max_depth:
                for link in links_from_page:
                    if link not in visited_urls and (link, current_depth + 1) not in crawl_queue:
                        crawl_queue.append((link, current_depth + 1))
                        pbar.total += 1 # زيادة العدد الكلي لشريط التقدم
                        pbar.refresh() # تحديث شريط التقدم فوراً

    print("\nعملية الزحف لـ HTML انتهت!")
    print(f"تم جمع {len(all_extracted_paragraphs)} فقرة نصية من HTML.")

    # حفظ جميع الفقرات المستخلصة في ملف
    with open(output_file, "w", encoding="utf-8") as f:
        for p in all_extracted_paragraphs:
            f.write(p + "\n")
    print(f"تم حفظ فقرات HTML الخام في الملف: {output_file}")


# --- الروابط الأولية للبدء بالزحف ---
initial_urls_for_html_crawl = [
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
    "https://shamuniversity.com/nav67"
]

# (تشغيل السكربت) ---
if __name__ == "__main__":
    output_html_file = "all_university_paragraphs.txt"
    crawl_html_website(initial_urls_for_html_crawl, max_depth=3, output_file=output_html_file)