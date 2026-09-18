"""
Receipt Analyzer — Analytics Dashboard

Run:
    streamlit run dashboard/app.py
"""
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACCURACY_FILE = 'data/processed/extraction_accuracy.json'
LABELS_FILE = 'data/processed/category_labels.json'
LAYOUTLM_FILE = 'data/processed/layoutlm_results.json'

st.set_page_config(
    page_title='Receipt Analyzer — Analytics',
    page_icon=None,
    layout='wide',
)

st.title('Receipt Analyzer — Analytics Dashboard')
st.caption('SROIE v2 dataset  |  pytesseract + TF-IDF + LogisticRegression  |  FastAPI + SQLite')


def load_accuracy_data():
    if not os.path.exists(ACCURACY_FILE):
        return None
    with open(ACCURACY_FILE, 'r') as f:
        return json.load(f)


def load_labels_data():
    if not os.path.exists(LABELS_FILE):
        return None
    with open(LABELS_FILE, 'r') as f:
        return json.load(f)


def load_layoutlm_data():
    if not os.path.exists(LAYOUTLM_FILE):
        return None
    with open(LAYOUTLM_FILE, 'r') as f:
        return json.load(f)


accuracy_data = load_accuracy_data()
labels_data = load_labels_data()
layoutlm_data = load_layoutlm_data()

with st.sidebar:
    st.header('Data Sources')
    acc_status = 'available' if accuracy_data is not None else 'missing'
    lbl_status = 'available' if labels_data is not None else 'missing'
    lm_status = 'available' if layoutlm_data is not None else 'missing (run evaluate_layoutlm.py)'
    st.write('extraction_accuracy.json:', acc_status)
    st.write('category_labels.json:', lbl_status)
    st.write('layoutlm_results.json:', lm_status)
    if accuracy_data is not None:
        n = len(accuracy_data.get('per_receipt', []))
        st.write('Test receipts evaluated:', n)
    if labels_data is not None:
        st.write('Training samples:', len(labels_data))

if accuracy_data is None and labels_data is None:
    st.error(
        'No data found. Run the following commands first:\n\n'
        '```\n'
        'python scripts/evaluate_extraction.py --split test\n'
        'python scripts/generate_category_labels.py\n'
        '```'
    )
    st.stop()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader('Field Extraction Accuracy')

    if accuracy_data is not None:
        summary = accuracy_data.get('summary', {})
        fields = ['company', 'date', 'address', 'total']
        rows = []
        for field in fields:
            field_data = summary.get(field, {})
            exact = round(field_data.get('exact_accuracy', 0.0) * 100, 1)
            f1 = round(field_data.get('avg_token_f1', 0.0) * 100, 1)
            rows.append({'Field': field, 'Exact %': exact, 'Token F1 %': f1})

        st.dataframe(rows, use_container_width=True, hide_index=True)

        chart_data = {}
        for row in rows:
            chart_data[row['Field']] = {'Exact %': row['Exact %'], 'Token F1 %': row['Token F1 %']}

        import pandas as pd
        df = pd.DataFrame(chart_data).T
        st.bar_chart(df, use_container_width=True)
    else:
        st.info('Run evaluate_extraction.py to see extraction accuracy.')

with col_right:
    st.subheader('Training Category Distribution')

    if labels_data is not None:
        counts = {}
        for item in labels_data:
            label = item.get('label', 'other')
            counts[label] = counts.get(label, 0) + 1

        sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))

        import pandas as pd
        df_cat = pd.DataFrame(
            {'Count': list(sorted_counts.values())},
            index=list(sorted_counts.keys()),
        )
        st.bar_chart(df_cat, use_container_width=True)

        st.dataframe(
            [{'Category': k, 'Count': v, 'Share %': round(v / len(labels_data) * 100, 1)}
             for k, v in sorted_counts.items()],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info('Run generate_category_labels.py to see category distribution.')

st.divider()

st.subheader('Per-Receipt F1 Score Distribution')

if accuracy_data is not None:
    per_receipt = accuracy_data.get('per_receipt', [])
    if per_receipt:
        import pandas as pd
        import numpy as np

        def compute_avg_f1(receipt_item):
            fields_data = receipt_item.get('fields', {})
            f1_values = []
            for fld in ['company', 'date', 'address', 'total']:
                fld_info = fields_data.get(fld, {})
                f1_values.append(fld_info.get('token_f1', 0.0))
            return sum(f1_values) / len(f1_values) if f1_values else 0.0

        avg_f1_scores = [compute_avg_f1(r) for r in per_receipt]

        bins = 10
        counts, bin_edges = np.histogram(avg_f1_scores, bins=bins, range=(0.0, 1.0))
        bin_labels = ['{:.0f}–{:.0f}%'.format(bin_edges[i] * 100, bin_edges[i + 1] * 100)
                      for i in range(len(counts))]

        df_hist = pd.DataFrame({'Receipts': counts}, index=bin_labels)
        st.bar_chart(df_hist, use_container_width=True)

        mean_f1 = round(float(np.mean(avg_f1_scores)) * 100, 1)
        median_f1 = round(float(np.median(avg_f1_scores)) * 100, 1)
        perfect = sum(1 for s in avg_f1_scores if s >= 1.0)
        zero = sum(1 for s in avg_f1_scores if s == 0.0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Mean F1', '{}%'.format(mean_f1))
        m2.metric('Median F1', '{}%'.format(median_f1))
        m3.metric('All 4 fields correct', str(perfect))
        m4.metric('0% score (OCR failure)', str(zero))

st.divider()

st.subheader('Sample Predictions vs Ground Truth')

if accuracy_data is not None:
    per_receipt = accuracy_data.get('per_receipt', [])
    if per_receipt:
        show_wrong_only = st.checkbox('Show only incorrect predictions', value=False)

        table_rows = []
        for r in per_receipt:
            fields_data = r.get('fields', {})
            for field in ['company', 'date', 'address', 'total']:
                field_info = fields_data.get(field, {})
                predicted = field_info.get('predicted', '')
                expected = field_info.get('expected', '')
                is_exact = field_info.get('exact_match', False)

                if show_wrong_only and is_exact:
                    continue

                table_rows.append({
                    'Receipt': r.get('stem', ''),
                    'Field': field,
                    'Predicted': str(predicted) if predicted else '',
                    'Expected': str(expected) if expected else '',
                    'Match': 'Yes' if is_exact else 'No',
                })

            if len(table_rows) >= 100:
                break

        if table_rows:
            import pandas as pd
            df_samples = pd.DataFrame(table_rows)
            st.dataframe(df_samples, use_container_width=True, hide_index=True)
        else:
            st.info('No rows to display with current filter.')

st.divider()

st.subheader('Field F1 Progress')

if accuracy_data is not None:
    summary = accuracy_data.get('summary', {})
    field_order = [
        ('date', 'Date'),
        ('total', 'Total'),
        ('company', 'Company'),
        ('address', 'Address'),
    ]
    for field_key, field_label in field_order:
        field_data = summary.get(field_key, {})
        f1 = field_data.get('avg_token_f1', 0.0)
        exact = field_data.get('exact_accuracy', 0.0)
        st.write('**{}** — F1: {:.1f}%  |  Exact: {:.1f}%'.format(field_label, f1 * 100, exact * 100))
        st.progress(f1)

st.divider()

st.subheader('Regex vs LayoutLM — Field Extraction Comparison')

if accuracy_data is None and layoutlm_data is None:
    st.info(
        'Run both evaluation scripts to see the comparison:\n\n'
        '```\n'
        'python scripts/evaluate_extraction.py\n'
        'python scripts/evaluate_layoutlm.py\n'
        '```'
    )
else:
    import pandas as pd

    fields_display = [
        ('company', 'Company'),
        ('date', 'Date'),
        ('address', 'Address'),
        ('total', 'Total'),
    ]

    regex_summary = accuracy_data.get('summary', {}) if accuracy_data is not None else {}
    lm_summary = layoutlm_data.get('summary', {}) if layoutlm_data is not None else {}

    comparison_rows = []
    for field_key, field_label in fields_display:
        regex_exact = regex_summary.get(field_key, {}).get('exact_accuracy', None)
        regex_f1 = regex_summary.get(field_key, {}).get('avg_token_f1', None)
        lm_exact = lm_summary.get(field_key, {}).get('exact_accuracy', None)
        lm_f1 = lm_summary.get(field_key, {}).get('avg_token_f1', None)

        row = {'Field': field_label}
        row['Regex Exact %'] = round(regex_exact * 100, 1) if regex_exact is not None else 'N/A'
        row['LayoutLM Exact %'] = round(lm_exact * 100, 1) if lm_exact is not None else 'N/A'
        if regex_exact is not None and lm_exact is not None:
            delta = round((lm_exact - regex_exact) * 100, 1)
            row['Delta pp'] = '+{}'.format(delta) if delta >= 0 else str(delta)
        else:
            row['Delta pp'] = 'N/A'
        row['Regex F1 %'] = round(regex_f1 * 100, 1) if regex_f1 is not None else 'N/A'
        row['LayoutLM F1 %'] = round(lm_f1 * 100, 1) if lm_f1 is not None else 'N/A'
        comparison_rows.append(row)

    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

    if regex_summary and lm_summary:
        chart_rows = {}
        for field_key, field_label in fields_display:
            regex_exact = regex_summary.get(field_key, {}).get('exact_accuracy', 0.0)
            lm_exact = lm_summary.get(field_key, {}).get('exact_accuracy', 0.0)
            chart_rows[field_label] = {
                'Regex Exact %': round(regex_exact * 100, 1),
                'LayoutLM Exact %': round(lm_exact * 100, 1),
            }
        df_compare = pd.DataFrame(chart_rows).T
        st.bar_chart(df_compare, use_container_width=True)
