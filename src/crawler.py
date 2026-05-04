"""
crawler.py — Thu thập nội dung bài báo từ danh sách URLs.

Sử dụng thư viện newspaper3k để trích xuất nội dung chính của trang web,
loại bỏ quảng cáo, menu, footer,... Hỗ trợ xử lý lỗi từng URL riêng lẻ
để không làm gián đoạn toàn bộ pipeline.
"""

import time
from typing import Optional
from newspaper import Article, ArticleException

from src.utils import logger


# ==============================================================================
# HÀM CRAWL ĐƠN LẺ
# ==============================================================================

def crawl_article(url: str, language: str = "vi", timeout: int = 10) -> Optional[str]:
    """
    Tải và trích xuất nội dung văn bản từ một URL đơn lẻ.

    Args:
        url: URL bài báo cần crawl
        language: Ngôn ngữ bài báo (mặc định 'vi' - tiếng Việt)
        timeout: Thời gian chờ tối đa (giây)

    Returns:
        Nội dung văn bản đã trích xuất, hoặc None nếu lỗi
    """
    try:
        logger.info(f"Đang crawl: {url}")

        article = Article(url, language=language, request_timeout=timeout)
        article.download()
        article.parse()

        text = article.text.strip()

        if not text:
            logger.warning(f"Không tìm thấy nội dung tại: {url}")
            return None

        word_count = len(text.split())
        logger.info(f"✅ Crawl thành công: {url} ({word_count} từ)")
        return text

    except ArticleException as e:
        logger.error(f"❌ Lỗi newspaper3k tại {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Lỗi không xác định tại {url}: {e}")
        return None


# ==============================================================================
# HÀM CRAWL NHIỀU URL
# ==============================================================================

def crawl_articles(
    urls: list[str],
    language: str = "vi",
    delay: float = 1.0,
    timeout: int = 10,
) -> list[str]:
    """
    Crawl danh sách URLs và trả về danh sách nội dung văn bản thành công.

    Mỗi URL được crawl tuần tự với độ trễ giữa các request để tránh
    bị chặn bởi server. Những URL lỗi sẽ bị bỏ qua.

    Args:
        urls: Danh sách URL cần crawl
        language: Ngôn ngữ (mặc định 'vi')
        delay: Thời gian chờ giữa các request (giây)
        timeout: Thời gian chờ mỗi request (giây)

    Returns:
        Danh sách chuỗi văn bản đã crawl thành công
    """
    if not urls:
        logger.warning("Danh sách URL rỗng, không có gì để crawl.")
        return []

    results = []
    total = len(urls)

    logger.info(f"Bắt đầu crawl {total} URL...")

    for i, url in enumerate(urls, start=1):
        logger.info(f"[{i}/{total}] Đang xử lý: {url}")

        text = crawl_article(url, language=language, timeout=timeout)
        if text:
            results.append(text)

        # Thêm delay giữa các request để tránh spam server
        if i < total:
            time.sleep(delay)

    success_count = len(results)
    fail_count = total - success_count
    logger.info(
        f"Crawl hoàn tất: {success_count}/{total} thành công, {fail_count} thất bại."
    )

    return results


# ==============================================================================
# HÀM GỘP VĂN BẢN
# ==============================================================================

def merge_texts(texts: list[str], separator: str = "\n\n") -> str:
    """
    Gộp nhiều đoạn văn bản thành một văn bản duy nhất.

    Args:
        texts: Danh sách các đoạn văn bản
        separator: Ký tự phân cách giữa các đoạn

    Returns:
        Văn bản đã gộp
    """
    cleaned = [t.strip() for t in texts if t and t.strip()]
    merged = separator.join(cleaned)
    logger.info(f"Đã gộp {len(cleaned)} đoạn văn bản ({len(merged.split())} từ).")
    return merged


# ==============================================================================
# CHẠY THỬ TRỰC TIẾP
# ==============================================================================

if __name__ == "__main__":
    # Ví dụ test với URL VnExpress
    test_urls = [
        "https://vnexpress.net/",
        "https://this-is-an-invalid-url.xyz/article",
    ]

    texts = crawl_articles(test_urls, delay=1.0)
    combined = merge_texts(texts)
    print("\n--- KẾT QUẢ CRAWL ---")
    print(combined[:500] if combined else "Không có nội dung.")
