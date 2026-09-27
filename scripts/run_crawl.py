import os
import re
import sys
from importlib import import_module

# ĐOẠN NÀY LÀ QUAN TRỌNG NHẤT:
# Nó ép Python phải nhìn vào thư mục gốc ai_job_server để thấy folder 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Dán link crawl trực tiếp vào đây nếu muốn chạy nhanh bằng file code.
# Có thể dán mỗi link một dòng, hoặc nhiều link ngăn cách bởi dấu phẩy/chấm phẩy.
DEFAULT_URLS_TEXT = """
https://topdev.vn/detail-jobs/lead-business-analyst-tester-erp-domain-biso24-2104884?src=topdev_home&medium=superhotjobs
"""

def _collect_urls_from_args() -> list[str]:
    raw_inputs = sys.argv[1:]
    if raw_inputs:
        merged = "\n".join(raw_inputs)
    elif DEFAULT_URLS_TEXT.strip():
        merged = DEFAULT_URLS_TEXT
    else:
        print("Dán danh sách URL rồi nhấn Ctrl+Z, Enter:")
        merged = sys.stdin.read()

    parts = re.split(r"[\r\n,;]+", merged)
    urls: list[str] = []
    seen: set[str] = set()
    for value in parts:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        urls.append(item)
    return urls


def start_crawling():
    session_module = import_module("app.db.session")
    crawler_module = import_module("app.services.crawler_service")

    get_standalone_db = getattr(session_module, "get_standalone_db")
    crawler_service = getattr(crawler_module, "CrawlerService")

    db = get_standalone_db()
    urls = _collect_urls_from_args()
    if not urls:
        print("Không có URL hợp lệ để crawl.")
        db.close()
        return

    results = crawler_service.crawl_jobs(db, urls)
    print(results)
    
    db.close()
    print("--- Xong ---")

if __name__ == "__main__":
    start_crawling()
