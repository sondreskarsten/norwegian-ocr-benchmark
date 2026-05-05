"""OCR-only scoring: numeric_recall, label_recall, distress_recall.

Extracted verbatim from notebooks/ocr_benchmark_11k.ipynb cell 8.
"""
import re

DISTRESS_PHRASES = [
    'fortsatt drift',
    'usikkerhet om fortsatt drift',
    'Egenkapitalen er tapt',
    'negativ egenkapital',
    'gjeldsforhandling',
]
NUM_RE = re.compile(r'-?\d+')


def number_present(n, text):
    if n is None:
        return False
    n = int(round(n))
    sign = '-' if n < 0 else ''
    n_abs = abs(n)
    s = str(n_abs)
    if n_abs == 0:
        return ' 0' in text or text.startswith('0') or '\n0' in text
    s_rev = s[::-1]
    groups = [s_rev[i:i+3][::-1] for i in range(0, len(s_rev), 3)][::-1]
    pat_grouped = sign + r'\s*'.join(re.escape(g) for g in groups)
    pat_digits  = sign + r'\s*'.join(re.escape(d) for d in s)
    return bool(re.search(pat_grouped, text)) or bool(re.search(pat_digits, text))


def collect_gemini_signal(gemini_json):
    nums, labels = set(), set()
    for arr in ('resultatregnskap', 'balanse_eiendeler',
                'balanse_egenkapital_og_gjeld', 'kontantstrom'):
        for item in (gemini_json.get(arr) or []):
            for k in ('amount_year', 'amount_prior_year'):
                v = item.get(k)
                if isinstance(v, (int, float)) and v != 0:
                    nums.add(int(round(v)))
            lab = (item.get('label') or '').strip()
            if len(lab) >= 4:
                labels.add(lab)
    for note in (gemini_json.get('noter') or []):
        title = (note.get('title') or '').strip()
        if len(title) >= 4:
            labels.add(title)
    distress = bool(gemini_json.get('going_concern_mentioned'))
    return nums, labels, distress


def score_ocr_text(ocr_text, gemini_json):
    nums, labels, distress_present = collect_gemini_signal(gemini_json)
    n_num = len(nums)
    n_num_hit = sum(1 for n in nums if number_present(n, ocr_text))
    n_lab = len(labels)
    text_lower = ocr_text.lower()
    n_lab_hit = sum(1 for l in labels if l.lower() in text_lower)
    if distress_present:
        n_distress_hit = sum(1 for ph in DISTRESS_PHRASES if ph.lower() in text_lower)
        distress_recall = n_distress_hit / len(DISTRESS_PHRASES)
    else:
        distress_recall = None
    ocr_nums = set()
    for tok in NUM_RE.findall(re.sub(r'(?<=\d)\s+(?=\d)', '', ocr_text)):
        try:
            v = int(tok)
            if abs(v) >= 1000:
                ocr_nums.add(v)
        except Exception:
            pass
    extra_in_ocr = len(ocr_nums - nums)
    return {
        'n_num_total': n_num, 'n_num_hit': n_num_hit,
        'numeric_recall': n_num_hit / n_num if n_num else None,
        'n_lab_total': n_lab, 'n_lab_hit': n_lab_hit,
        'label_recall': n_lab_hit / n_lab if n_lab else None,
        'n_chars_ocr': len(ocr_text),
        'n_extra_numbers_in_ocr': extra_in_ocr,
        'distress_present_in_gemini': distress_present,
        'distress_recall': distress_recall,
    }
