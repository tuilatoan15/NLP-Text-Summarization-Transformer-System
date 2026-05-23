from src.preprocess import (
    clean_text,
    fix_decimal_spacing,
    is_editorial_noise_sentence,
    strip_editorial_chrome,
)


def test_fix_decimal_spacing():
    assert fix_decimal_spacing("điểm 4. 0") == "điểm 4.0"
    assert fix_decimal_spacing("9, 25") == "9,25"


def test_strip_photo_captions():
    raw = """5 sinh viên tốt nghiệp sớm với điểm 4.0
13
Năm sinh viên đạt GPA tuyệt đối 4.0.
Trần Thế Vũ, tốt nghiệp ngành Kinh doanh quốc tế. Ảnh: Nhân vật cung cấp
1 / 5
Các sinh viên cho biết phương pháp chủ yếu là chuẩn bị bài."""
    cleaned = strip_editorial_chrome(raw)
    assert "Ảnh:" not in cleaned
    assert "13" not in cleaned.split()
    assert "1 / 5" not in cleaned


def test_is_editorial_noise_sentence():
    assert is_editorial_noise_sentence("Trần Thế Vũ, tốt nghiệp ngành X. Ảnh: Nhân vật cung cấp")
    assert not is_editorial_noise_sentence(
        "Năm sinh viên đạt GPA tuyệt đối 4.0 sau ba đợt xét tốt nghiệp sớm."
    )


def test_aggressive_clean_keeps_gpa_decimal():
    text = clean_text("Sinh viên đạt điểm 4.0 và GPA 9,25.", aggressive=True)
    assert "4.0" in text
    assert "9,25" in text
