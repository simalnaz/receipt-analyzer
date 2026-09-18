import re
from app.models.schemas import ReceiptFields

COMPANY_SEARCH_DEPTH = 10

DATE_PATTERNS = [
    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
    r'\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
    r'\b(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4})\b',
]

TOTAL_PATTERNS = [
    r'(?:GRAND\s*TOTAL|JUMLAH\s*BESAR|AMT\s*DUE)[^\d\n]*(\d+[\.,]\d{2})',
    r'NET\s*(?:TOTAL|AMOUNT)[^\d\n]*(\d+[\.,]\d{2})',
    r'(?:^|\n)\s*TOTAL(?!\s*(?:ITEM|TAX|QTY|PCS|DISC|PAGE))[^\d\n]*(\d+[\.,]\d{2})',
    r'(?:AMAUN|AMOUNT|JUMLAH)[^\d\n]*(\d+[\.,]\d{2})',
    r'(?:RM|MYR)\s*(\d+[\.,]\d{2})\s*$',
]

ADDRESS_KEYWORDS = re.compile(
    r'\b(NO\.?|NO\s+\d|JALAN|JLN|LOT|FLOOR|LEVEL|STREET|ST\.|AVE|ROAD|RD\.|'
    r'BLOK|BLOCK|TINGKAT|TAMAN|KOMPLEKS|PUSAT|BANDAR|LORONG|LRG|KG\.?|KAMPUNG|'
    r'DESA|PERSIARAN|LEBUH|LEBUHRAY A|PLAZA|MENARA)\b'
)

COMPANY_SIGNALS = re.compile(
    r'\b(SDN\.?\s*BHD\.?|BHD\.?|PTE\.?\s*LTD\.?|LTD\.?|LLC|INC\.?|CORP\.?|'
    r'SDN|ENTERPRISE|TRADING|HOLDINGS|GROUP|MARKETING|RESTAURANT|INDUSTRIES|'
    r'SUPERMARKET|HYPERMARKET|PHARMACY|HARDWARE|BAKERY|SERVICES|SUPPLY|'
    r'GLOBAL|INTERNATIONAL|RESOURCES|MANAGEMENT|TECHNOLOGY)\b'
)

NOISE_WORDS = {
    'receipt', 'invoice', 'tax', 'official', 'copy', 'bill',
    'thank', 'you', 'please', 'come', 'again', 'welcome',
    'customer', 'cashier', 'operator',
}

NOISE_LINE_PATTERN = re.compile(r'^[#\*\-=\s]+$')


def _is_noise_line(line):
    lower = line.lower()
    if NOISE_LINE_PATTERN.match(line):
        return True
    word_set = set(re.findall(r'[a-z]+', lower))
    if word_set and word_set.issubset(NOISE_WORDS):
        return True
    return False


def extract_date_from_text(text):
    upper_text = text.upper()
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, upper_text)
        if match is not None:
            return match.group(1)
    return None


def _parse_amount_string(raw_value):
    cleaned = raw_value.strip()
    # handle European format: 1.234,56 → 1234.56
    if re.search(r'\d{1,3}\.\d{3},\d{2}$', cleaned) is not None:
        cleaned = cleaned.replace('.', '').replace(',', '.')
    else:
        cleaned = cleaned.replace(',', '.')
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_total_from_text(text):
    upper_text = text.upper()
    for pattern in TOTAL_PATTERNS:
        match = re.search(pattern, upper_text)
        if match is not None:
            value = _parse_amount_string(match.group(1))
            if value is not None:
                return value
    return None


def _truncate_at_signal_end(line):
    match = COMPANY_SIGNALS.search(line.upper())
    if match is None:
        return line
    end = match.end()
    # include the matched signal word and stop there
    candidate = line[:end].strip()
    return candidate if len(candidate) >= 3 else line


def _has_sufficient_alphanumeric(text, min_ratio=0.6):
    if len(text) == 0:
        return False
    count = len(re.findall(r'[a-zA-Z0-9]', text))
    return (count / len(text)) >= min_ratio


def extract_company_from_text(lines):
    candidate_with_signal = None
    candidate_fallback = None

    for line in lines[:COMPANY_SEARCH_DEPTH]:
        cleaned = line.strip()
        if len(cleaned) < 3:
            continue
        if _is_noise_line(cleaned):
            continue
        if not _has_sufficient_alphanumeric(cleaned):
            continue

        if COMPANY_SIGNALS.search(cleaned.upper()) is not None:
            truncated = _truncate_at_signal_end(cleaned)
            if re.search(r'\d{3,}', truncated) is None and candidate_with_signal is None:
                if _has_sufficient_alphanumeric(truncated):
                    candidate_with_signal = truncated
        else:
            if re.search(r'\d{3,}', cleaned) is None and candidate_fallback is None:
                candidate_fallback = cleaned

    if candidate_with_signal is not None:
        return candidate_with_signal
    return candidate_fallback


def _find_postcode_line_index(lines):
    for i, line in enumerate(lines):
        if re.search(r'\b\d{5}\b', line) is not None:
            return i
    return None


def extract_address_from_text(lines):
    search_lines = lines[1:20]

    # Strategy 1: find postcode line and walk backwards to collect address block
    postcode_idx = _find_postcode_line_index(search_lines)
    if postcode_idx is not None:
        address_lines = []
        start = max(0, postcode_idx - 3)
        for line in search_lines[start:postcode_idx + 1]:
            cleaned = line.strip()
            if cleaned != '':
                address_lines.append(cleaned)
        if address_lines:
            return ', '.join(address_lines)

    # Strategy 2: keyword anchor then collect continuation lines
    address_lines = []
    address_started = False

    for line in search_lines:
        cleaned = line.strip()
        if cleaned == '':
            if address_started:
                break
            continue

        if ADDRESS_KEYWORDS.search(cleaned.upper()) is not None:
            address_started = True
            address_lines.append(cleaned)
            continue

        if address_started:
            address_lines.append(cleaned)
            if len(address_lines) >= 4:
                break

    if not address_lines:
        return None
    return ', '.join(address_lines)


def extract_receipt_fields(raw_text):
    lines = []
    for line in raw_text.splitlines():
        if line.strip() != '':
            lines.append(line)

    return ReceiptFields(
        company=extract_company_from_text(lines),
        address=extract_address_from_text(lines),
        date=extract_date_from_text(raw_text),
        total=extract_total_from_text(raw_text),
    )
