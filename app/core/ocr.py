import pytesseract
from PIL import Image
from app.utils.preprocessing import prepare_image_for_ocr

DEFAULT_TESSERACT_CONFIG = '--oem 3 --psm 4'


def extract_text_from_image(image_path, tesseract_config=None, preprocessing_config=None):
    if tesseract_config is None:
        tesseract_config = DEFAULT_TESSERACT_CONFIG
    processed = prepare_image_for_ocr(image_path, config=preprocessing_config)
    pil_image = Image.fromarray(processed)
    raw_text = pytesseract.image_to_string(pil_image, config=tesseract_config)
    return raw_text.strip()


def extract_text_and_confidence(image_path, tesseract_config=None, preprocessing_config=None):
    if tesseract_config is None:
        tesseract_config = DEFAULT_TESSERACT_CONFIG
    processed = prepare_image_for_ocr(image_path, config=preprocessing_config)
    pil_image = Image.fromarray(processed)
    data = pytesseract.image_to_data(
        pil_image,
        config=tesseract_config,
        output_type=pytesseract.Output.DICT,
    )

    lines = {}
    words = []

    for i in range(len(data['text'])):
        word = data['text'][i]
        if word.strip() == '':
            continue
        confidence = int(data['conf'][i])
        line_num = data['line_num'][i]
        words.append({
            'word': word,
            'confidence': confidence,
            'line': line_num,
        })
        if line_num not in lines:
            lines[line_num] = []
        lines[line_num].append(word)

    full_text = '\n'.join(
        ' '.join(lines[k]) for k in sorted(lines.keys())
    )
    return full_text, words


def compute_ocr_confidence(word_data):
    if not word_data:
        return 0.0
    total = 0.0
    count = 0
    for w in word_data:
        if w['confidence'] >= 0:
            total += w['confidence']
            count += 1
    if count == 0:
        return 0.0
    return total / count


def flag_low_confidence_words(word_data, confidence_threshold=60):
    result = []
    for w in word_data:
        if 0 <= w['confidence'] < confidence_threshold:
            result.append(w)
    return result
