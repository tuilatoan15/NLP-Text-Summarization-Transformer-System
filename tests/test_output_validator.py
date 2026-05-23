"""Tests for multilingual garbage detection in generated summaries."""

from src.output_validator import is_garbled_abstractive, is_multilingual_garbage, validate_output


def test_detects_mt5_style_garbage():
    garbage = "a улан lytte ෝර koske"
    assert is_multilingual_garbage(garbage, require_vietnamese=True)
    result = validate_output(garbage, require_vietnamese=True)
    assert result["is_corrupted"]


def test_detects_vit5_garbled_output():
    garbage = (
        "nhân vật cung cấp Trần Thị Châu Anh, tốt nghiệp ngành Luật kinh tế. "
        "WỴ tốt nghiệp trường Luật kinh tê ngành Luật tốt nghiệp đại học"
    )
    assert is_garbled_abstractive(garbage)
    assert validate_output(garbage)["is_corrupted"]


def test_accepts_vietnamese_summary():
    text = (
        "Tập đoàn Điện lực Việt Nam cho biết nhu cầu tiêu thụ điện "
        "trong mùa nắng nóng tiếp tục tăng cao tại nhiều địa phương."
    )
    assert not is_multilingual_garbage(text, require_vietnamese=True)
    assert not validate_output(text, require_vietnamese=True)["is_corrupted"]
