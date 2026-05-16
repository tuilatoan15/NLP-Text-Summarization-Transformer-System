import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path

def collect_vnexpress_links(limit=100):
    print(f"--- Đang thu thập {limit} link từ VnExpress ---")
    base_url = "https://vnexpress.net/tin-tuc-24h"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    links = []
    page = 1
    
    while len(links) < limit:
        url = f"{base_url}/p{page}"
        print(f"Đang quét trang {page}...")
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            
            # VnExpress news items are usually in .title-news a
            items = soup.select(".title-news a")
            if not items:
                print("Không tìm thấy link nào thêm. Kết thúc.")
                break
                
            for item in items:
                link = item.get("href")
                link = str(link) if link is not None else None
                if link and link.startswith("https://vnexpress.net/") and link.endswith(".html"):
                    if link not in links:
                        links.append(link)
                        if len(links) >= limit:
                            break
            
            page += 1
            time.sleep(0.5) # Be nice
        except Exception as e:
            print(f"Lỗi: {e}")
            break
            
    return links[:limit]

if __name__ == "__main__":
    links = collect_vnexpress_links(100)
    
    output_path = Path("storage/collected_links.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"links": links, "count": len(links)}, f, indent=2)
        
    print(f"\n✅ Đã thu thập {len(links)} links.")
    print(f"Kết quả lưu tại: {output_path}")
    
    # In ra 5 link đầu tiên để xem thử
    for l in links[:5]:
        print(f" - {l}")
