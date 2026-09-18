import torch
import pytesseract
from PIL import Image
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

from app.models.schemas import ReceiptFields
from app.utils.preprocessing import prepare_image_for_ocr

DEFAULT_LAYOUTLM_CONFIG = {
    'model_path': 'data/layoutlm_finetuned',
    'max_seq_length': 512,
    'tesseract_config': '--oem 3 --psm 4',
}

LABEL2ID = {
    'O': 0,
    'B-COMPANY': 1, 'I-COMPANY': 2,
    'B-DATE': 3,    'I-DATE': 4,
    'B-ADDRESS': 5, 'I-ADDRESS': 6,
    'B-TOTAL': 7,   'I-TOTAL': 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

_model = None
_tokenizer = None


def _load_model(model_path):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
        _model = LayoutLMForTokenClassification.from_pretrained(model_path)
        _model.eval()
    return _model, _tokenizer


def _extract_words_and_boxes_from_image(image_path, tesseract_config):
    """
    Run Tesseract and return word-level bounding boxes normalized to [0, 1000].
    """
    processed = prepare_image_for_ocr(image_path)
    pil_image = Image.fromarray(processed)
    img_width, img_height = pil_image.size

    data = pytesseract.image_to_data(
        pil_image,
        config=tesseract_config,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    boxes = []

    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if not word:
            continue
        if int(data['conf'][i]) < 0:
            continue
        left   = data['left'][i]
        top    = data['top'][i]
        width  = data['width'][i]
        height = data['height'][i]
        x0 = max(0, min(1000, int(left / img_width * 1000)))
        y0 = max(0, min(1000, int(top / img_height * 1000)))
        x1 = max(0, min(1000, int((left + width) / img_width * 1000)))
        y1 = max(0, min(1000, int((top + height) / img_height * 1000)))
        words.append(word)
        boxes.append([x0, y0, x1, y1])

    return words, boxes


def _predict_word_labels(model, tokenizer, words, boxes, max_seq_length):
    """
    Tokenize words, run the model, and return a label ID per word.
    Uses the first sub-token's predicted label for each word.
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt',
    )

    word_ids = encoding.word_ids()

    aligned_boxes = []
    for word_id in word_ids:
        if word_id is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_id])

    bbox_tensor = torch.tensor([aligned_boxes], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            token_type_ids=encoding['token_type_ids'],
            bbox=bbox_tensor,
        )

    predictions = outputs.logits[0].argmax(dim=-1).tolist()

    word_labels = [None] * len(words)
    prev_word_id = None

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            prev_word_id = None
            continue
        if word_id != prev_word_id:
            word_labels[word_id] = predictions[token_idx]
        prev_word_id = word_id

    for i in range(len(word_labels)):
        if word_labels[i] is None:
            word_labels[i] = LABEL2ID['O']

    return word_labels


def _group_words_by_field(words, word_labels):
    """
    Collect consecutive words that belong to the same entity span.
    Returns a dict: {'COMPANY': str, 'DATE': str, 'ADDRESS': str, 'TOTAL': str}
    """
    field_tokens = {'COMPANY': [], 'DATE': [], 'ADDRESS': [], 'TOTAL': []}

    for word, label_id in zip(words, word_labels):
        label = ID2LABEL.get(label_id, 'O')
        if label == 'O':
            continue
        bio, field = label.split('-', 1)
        if field in field_tokens:
            field_tokens[field].append(word)

    return {field: ' '.join(tokens) if tokens else None
            for field, tokens in field_tokens.items()}


def _parse_total(value_str):
    if not value_str:
        return None
    cleaned = value_str.upper()
    for prefix in ['RM', 'MYR', 'USD', '$']:
        cleaned = cleaned.replace(prefix, '')
    cleaned = cleaned.strip()
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def extract_fields_from_words(words, boxes, config=None):
    """
    Extract receipt fields from pre-computed words and normalized [0, 1000] boxes.
    Used for evaluation on SROIE box files (no Tesseract step).

    Returns a ReceiptFields instance.
    """
    if config is None:
        config = DEFAULT_LAYOUTLM_CONFIG

    model_path = config.get('model_path', DEFAULT_LAYOUTLM_CONFIG['model_path'])
    max_seq_length = config.get('max_seq_length', DEFAULT_LAYOUTLM_CONFIG['max_seq_length'])

    model, tokenizer = _load_model(model_path)

    if not words:
        return ReceiptFields()

    word_labels = _predict_word_labels(model, tokenizer, words, boxes, max_seq_length)
    field_strings = _group_words_by_field(words, word_labels)

    return ReceiptFields(
        company=field_strings['COMPANY'] or None,
        address=field_strings['ADDRESS'] or None,
        date=field_strings['DATE'] or None,
        total=_parse_total(field_strings['TOTAL']),
    )


def extract_fields_with_layoutlm(image_path, config=None):
    """
    Extract receipt fields (company, address, date, total) from an image
    using the fine-tuned LayoutLM model.

    Returns a ReceiptFields instance — same type as extract_receipt_fields().
    Raises FileNotFoundError if the fine-tuned model is not found.
    """
    if config is None:
        config = DEFAULT_LAYOUTLM_CONFIG

    model_path = config.get('model_path', DEFAULT_LAYOUTLM_CONFIG['model_path'])
    max_seq_length = config.get('max_seq_length', DEFAULT_LAYOUTLM_CONFIG['max_seq_length'])
    tesseract_config = config.get('tesseract_config', DEFAULT_LAYOUTLM_CONFIG['tesseract_config'])

    model, tokenizer = _load_model(model_path)

    words, boxes = _extract_words_and_boxes_from_image(image_path, tesseract_config)

    if not words:
        return ReceiptFields()

    word_labels = _predict_word_labels(model, tokenizer, words, boxes, max_seq_length)
    field_strings = _group_words_by_field(words, word_labels)

    return ReceiptFields(
        company=field_strings['COMPANY'] or None,
        address=field_strings['ADDRESS'] or None,
        date=field_strings['DATE'] or None,
        total=_parse_total(field_strings['TOTAL']),
    )
