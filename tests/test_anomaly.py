import pytest
from app.core.anomaly import (
    detect_amount_out_of_range,
    detect_ocr_garbage_characters,
    detect_missing_critical_fields,
    detect_statistical_amount_outlier,
    detect_receipt_anomalies,
)


def test_detect_amount_above_threshold():
    assert detect_amount_out_of_range(15000.0) is True


def test_detect_amount_below_minimum():
    assert detect_amount_out_of_range(0.001) is True


def test_normal_amount_is_not_flagged():
    assert detect_amount_out_of_range(45.50) is False


def test_detect_garbage_characters_in_heavy_noise():
    noisy_text = 'ABC\x00\x01\x02\x03\x04' * 10
    assert detect_ocr_garbage_characters(noisy_text) is True


def test_clean_text_has_no_garbage():
    assert detect_ocr_garbage_characters('TOTAL RM 10.00') is False


def test_detect_missing_two_or_more_critical_fields():
    assert detect_missing_critical_fields(None, None, 10.0) is True
    assert detect_missing_critical_fields(None, None, None) is True
    assert detect_missing_critical_fields('Company', None, None) is True


def test_all_fields_present_is_not_missing():
    assert detect_missing_critical_fields('Company', '2024-01-01', 50.0) is False


def test_statistical_outlier_detected():
    historical = [10.0, 12.0, 11.5, 9.8, 10.5, 11.0]
    assert detect_statistical_amount_outlier(500.0, historical) is True


def test_normal_value_not_statistical_outlier():
    historical = [10.0, 12.0, 11.5, 9.8, 10.5, 11.0]
    assert detect_statistical_amount_outlier(11.0, historical) is False


def test_full_anomaly_pipeline_with_clean_receipt():
    report = detect_receipt_anomalies(
        raw_text='SHOP ABC\nDate: 01/01/2024\nTOTAL RM 25.50',
        company='SHOP ABC',
        date='01/01/2024',
        total=25.50,
    )
    assert report.has_anomaly is False
    assert report.anomaly_types == []
