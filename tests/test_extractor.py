from app.core.extractor import (
    extract_date_from_text,
    extract_total_from_text,
    extract_company_from_text,
    extract_receipt_fields,
)


def test_extract_date_finds_standard_format(sample_receipt_text):
    assert extract_date_from_text(sample_receipt_text) == '15/06/2024'


def test_extract_total_finds_grand_total(sample_receipt_text):
    total = extract_total_from_text(sample_receipt_text)
    assert total == 21.40


def test_extract_company_returns_first_meaningful_line(sample_lines):
    company = extract_company_from_text(sample_lines)
    assert company == 'KEDAI RUNCIT ABC SDN BHD'


def test_extract_receipt_fields_returns_all_fields(sample_receipt_text):
    fields = extract_receipt_fields(sample_receipt_text)
    assert fields.company is not None
    assert fields.date == '15/06/2024'
    assert fields.total == 21.40


def test_extract_date_returns_none_when_absent():
    assert extract_date_from_text('No date here') is None


def test_extract_total_returns_none_when_absent():
    assert extract_total_from_text('No amount here') is None


def test_company_truncates_at_signal_word():
    lines = ['GOLDEN ARCHES SDN BHD Tax Invoice Cashier: 001']
    assert extract_company_from_text(lines) == 'GOLDEN ARCHES SDN BHD'


def test_company_skips_noise_only_lines():
    lines = ['OFFICIAL RECEIPT', 'PERNIAGAAN ZHENG HUI SDN BHD', '47500 Kuala Lumpur']
    company = extract_company_from_text(lines)
    assert 'ZHENG HUI' in company
