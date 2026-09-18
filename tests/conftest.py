import pytest

SAMPLE_RECEIPT_TEXT = """
KEDAI RUNCIT ABC SDN BHD
No. 12, Jalan Mawar 3
Taman Bunga, 47500 Subang Jaya

Date: 15/06/2024
Time: 14:32

Item 1                  RM 5.50
Item 2                  RM 12.00
Item 3                  RM 3.90

TOTAL                   RM 21.40
Thank you for shopping!
"""

SAMPLE_LINES = [l.strip() for l in SAMPLE_RECEIPT_TEXT.splitlines() if l.strip() != '']


@pytest.fixture
def sample_receipt_text():
    return SAMPLE_RECEIPT_TEXT


@pytest.fixture
def sample_lines():
    return SAMPLE_LINES
