import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import fix_spaced_letters, detect_garbled_text

test_text = "r ên ứng d ụng di đ ụng sinh vi ên c ó th ểt ương t ác v à th ực hi ện c ác ch ức n ăng nh ư đ ăng k ý"
print(f"Original: {test_text}")
print(f"Garbled detected: {detect_garbled_text(test_text)}")
fixed = fix_spaced_letters(test_text)
print(f"Fixed: {fixed}")

test_text_2 = "Xửl ý d ữli ệu nhanh ch óng v à c ó th ểáp d ụng g ần th ời gian th ực"
print(f"\nOriginal 2: {test_text_2}")
fixed_2 = fix_spaced_letters(test_text_2)
print(f"Fixed 2: {fixed_2}")

test_text_3 = "MỤ C LỤ C!Y>ỘjỘ&ẬPW THIỆỴ THIỆẼỘỠ"
print(f"\nOriginal 3: {test_text_3}")
fixed_3 = fix_spaced_letters(test_text_3)
print(f"Fixed 3: {fixed_3}")
