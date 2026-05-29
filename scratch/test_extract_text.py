import requests
import sys

# Force UTF-8 output encoding for standard output
sys.stdout.reconfigure(encoding='utf-8')

def test_extract_txt():
    url = "http://localhost:8000/extract-text"
    
    # Create a dummy txt file
    dummy_content = "Trí tuệ nhân tạo đang phát triển mạnh mẽ và thay đổi thế giới."
    files = {
        "file": ("test.txt", dummy_content.encode('utf-8'), "text/plain")
    }
    
    try:
        response = requests.post(url, files=files)
        print("Status code:", response.status_code)
        if response.status_code == 200:
            data = response.json()
            print("Extracted text (length):", len(data.get("text", "")))
            print("Filename:", data.get("filename"))
            print("Word count:", data.get("word_count"))
            print("Success:", data.get("success"))
            assert data.get("success") is True
            assert "Trí tuệ nhân tạo" in data.get("text")
            print("Success status confirmed. Text extraction works perfectly!")
        else:
            print("Failed:", response.status_code)
    except Exception as e:
        print("Error during request:", str(e))

if __name__ == "__main__":
    test_extract_txt()
