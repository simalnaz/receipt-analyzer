"""
Fine-tune LayoutLM for token classification (NER) on SROIE receipts.

Loads the BIO-labeled dataset from prepare_dataset.py, fine-tunes
LayoutLMForTokenClassification with a manual training loop (no HuggingFace Trainer),
and saves the model to data/layoutlm_finetuned/.

Usage:
    python scripts/train_layoutlm.py
    python scripts/train_layoutlm.py --epochs 5 --batch_size 8
    python scripts/train_layoutlm.py --epochs 10 --lr 3e-5
"""
import argparse
import json
import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRETRAINED_MODEL_PATH = 'data/sroie/SROIE2019/layoutlm-base-uncased'
DATASET_DIR = 'data/processed/layoutlm_dataset'
OUTPUT_DIR = 'data/layoutlm_finetuned'
MAX_SEQ_LENGTH = 512

LABEL2ID = {
    'O': 0,
    'B-COMPANY': 1, 'I-COMPANY': 2,
    'B-DATE': 3,    'I-DATE': 4,
    'B-ADDRESS': 5, 'I-ADDRESS': 6,
    'B-TOTAL': 7,   'I-TOTAL': 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SROIEDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.encodings = [tokenize_and_align(tokenizer, ex) for ex in examples]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {
            'input_ids':      torch.tensor(item['input_ids'],      dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
            'bbox':           torch.tensor(item['bbox'],           dtype=torch.long),
            'labels':         torch.tensor(item['labels'],         dtype=torch.long),
        }


def tokenize_and_align(tokenizer, example):
    """
    Tokenize words and align labels + bounding boxes to sub-tokens.

    First sub-token of each word inherits the word's label and bbox.
    Continuation sub-tokens and special tokens get label -100 (ignored in loss).
    Padding tokens get label -100 and bbox [0, 0, 0, 0].
    """
    encoding = tokenizer(
        example['words'],
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    word_ids = encoding.word_ids()
    aligned_labels = []
    aligned_boxes = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
            aligned_boxes.append([0, 0, 0, 0])
        elif word_id != prev_word_id:
            aligned_labels.append(example['labels'][word_id])
            aligned_boxes.append(example['boxes'][word_id])
        else:
            aligned_labels.append(-100)
            aligned_boxes.append(example['boxes'][word_id])
        prev_word_id = word_id

    encoding['labels'] = aligned_labels
    encoding['bbox'] = aligned_boxes
    return encoding


def load_pretrained_with_remapping(model_path):
    """
    Local checkpoint uses bert.* prefix (transformers 4.x format).
    Transformers 5.x expects layoutlm.* — remap keys before loading.
    Classifier head is absent from the pretrained model; skipped with strict=False (random init).
    """
    from transformers import LayoutLMConfig
    config = LayoutLMConfig.from_pretrained(model_path)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID

    model = LayoutLMForTokenClassification(config)

    checkpoint_path = os.path.join(model_path, 'pytorch_model.bin')
    raw_sd = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    remapped = {}
    for key, value in raw_sd.items():
        if key.startswith('bert.'):
            remapped['layoutlm.' + key[len('bert.'):]] = value
        # cls.* MLM head — not needed for token classification, skip

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing_non_classifier = [k for k in missing if not k.startswith('classifier')]
    if missing_non_classifier:
        print('WARNING: Unexpected missing keys: {}'.format(missing_non_classifier))
    if unexpected:
        print('WARNING: Unexpected keys in checkpoint: {}'.format(unexpected))

    print('Pretrained weights loaded ({} keys remapped, classifier head randomly initialized)'.format(
        len(remapped)))
    return model


def load_examples(split):
    path = os.path.join(DATASET_DIR, '{}.json'.format(split))
    if not os.path.exists(path):
        print('ERROR: {} not found. Run prepare_dataset.py first.'.format(path))
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_epoch(model, loader, optimizer, device, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            bbox           = batch['bbox'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                labels=labels,
            )

            loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def train(epochs, batch_size, lr):
    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )
    print('Device: {}'.format(device))

    print('Loading tokenizer from {} ...'.format(PRETRAINED_MODEL_PATH))
    tokenizer = LayoutLMTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)

    print('Loading train examples ...')
    train_examples = load_examples('train')
    print('Loading test examples (used as validation) ...')
    val_examples = load_examples('test')
    print('Train: {}  Val: {}'.format(len(train_examples), len(val_examples)))

    print('Tokenizing and aligning labels ...')
    train_dataset = SROIEDataset(train_examples, tokenizer)
    val_dataset = SROIEDataset(val_examples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('Loading LayoutLM from {} ...'.format(PRETRAINED_MODEL_PATH))
    model = load_pretrained_with_remapping(PRETRAINED_MODEL_PATH)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    print('\nStarting training ({} epochs, batch_size={}, lr={}) ...\n'.format(
        epochs, batch_size, lr))

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, is_train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, is_train=False)

        marker = '  <-- best' if val_loss < best_val_loss else ''
        print('Epoch {:>2}/{}: train_loss={:.4f}  val_loss={:.4f}{}'.format(
            epoch, epochs, train_loss, val_loss, marker))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    print('\nTraining complete. Best val_loss={:.4f}'.format(best_val_loss))
    print('Model saved to {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune LayoutLM on SROIE.')
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=5e-5)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
