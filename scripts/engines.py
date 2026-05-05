"""Engine factories — copied byte-for-byte from notebooks/ocr_benchmark_11k.ipynb.

Do not modify these in this file. Each factory installs deps lazily on first
call and returns a callable fn(orgnr, bundle) -> dict.
"""
import os
import subprocess


def _pip(pkg):
    subprocess.run(['pip', 'install', '-q', *pkg.split()], check=False)


def _apt(pkgs):
    subprocess.run(['apt-get', '-qq', 'install', '-y', *pkgs.split()],
                   check=False, capture_output=True)


# === lightweight CPU OCR engines ===

def make_tesseract():
    _apt('tesseract-ocr tesseract-ocr-nor')
    _pip('pytesseract')
    import pytesseract
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            txt = pytesseract.image_to_string(p, lang='nor')
            pages.append({'text': txt})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_easyocr():
    _pip('easyocr')
    import easyocr, torch
    reader = easyocr.Reader(['no', 'en'], gpu=torch.cuda.is_available())
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            res = reader.readtext(p, detail=0, paragraph=True)
            pages.append({'text': '\n'.join(res)})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_paddleocr():
    # paddleocr>=3.x pulls in langchain_text_splitters; pin to 2.7.x which
    # uses the classic API that the factory below targets.
    _pip('paddleocr==2.7.3 paddlepaddle-gpu==2.6.1')
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            r = ocr.ocr(p, cls=False)
            lines = []
            if r and r[0]:
                for line in r[0]:
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        lines.append(line[1][0])
            pages.append({'text': '\n'.join(lines)})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_ocrmypdf():
    _apt('ocrmypdf tesseract-ocr-nor')
    _pip('pdfplumber')
    import pdfplumber
    def fn(orgnr, b):
        out_pdf = b['pdf'].replace('.pdf', '_ocr.pdf')
        subprocess.run(['ocrmypdf', '--force-ocr', '-l', 'nor', b['pdf'], out_pdf],
                       check=False, capture_output=True)
        if not os.path.exists(out_pdf):
            return {'full_text': '', 'pages': [], 'n_pages': 0,
                    'error': 'ocrmypdf failed'}
        with pdfplumber.open(out_pdf) as pdf:
            pages = [{'text': pg.extract_text() or ''} for pg in pdf.pages]
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


# === GPU-friendly OCR / vision engines ===

def make_doctr():
    _pip('python-doctr[torch]')
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    model = ocr_predictor(pretrained=True)
    def fn(orgnr, b):
        doc = DocumentFile.from_pdf(b['pdf'])
        res = model(doc)
        pages = []
        for page in res.pages:
            lines = []
            for block in page.blocks:
                for line in block.lines:
                    lines.append(' '.join(w.value for w in line.words))
            pages.append({'text': '\n'.join(lines)})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_nougat():
    # Pin albumentations<2 — newer versions enforce pydantic literal validation
    # on compression_type which rejects nougat's int default.
    _pip('nougat-ocr "albumentations<2.0" "pydantic<2.10"')
    import torch
    from nougat import NougatModel
    from nougat.utils.checkpoint import get_checkpoint
    from nougat.postprocessing import markdown_compatible
    ckpt = get_checkpoint('0.1.0-base')
    model = NougatModel.from_pretrained(ckpt)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    def fn(orgnr, b):
        from PIL import Image
        pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            pv = model.encoder.prepare_input(img, random_padding=False).unsqueeze(0)
            if torch.cuda.is_available():
                pv = pv.cuda()
            with torch.no_grad():
                out = model.inference(image_tensors=pv)
            md = markdown_compatible(out['predictions'][0])
            pages.append({'text': md})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_trocr():
    _pip('transformers')
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    if torch.cuda.is_available():
        model = model.cuda()
    def fn(orgnr, b):
        pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            pv = proc(images=img, return_tensors='pt').pixel_values
            if torch.cuda.is_available():
                pv = pv.cuda()
            with torch.no_grad():
                ids = model.generate(pv, max_length=512)
            txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
            pages.append({'text': txt})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_surya():
    # Pin to 0.6.13 — newer surya-ocr (>=0.7) reorganized modules and removed
    # surya.ocr / surya.model.detection / surya.model.recognition.
    _pip('surya-ocr==0.6.13')
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det, load_processor as load_det_proc
    from surya.model.recognition.model import load_model as load_rec
    from surya.model.recognition.processor import load_processor as load_rec_proc
    from PIL import Image
    det_m, det_p = load_det(), load_det_proc()
    rec_m, rec_p = load_rec(), load_rec_proc()
    def fn(orgnr, b):
        imgs = [Image.open(p).convert('RGB') for p in b['page_imgs']]
        langs = [['no', 'en']] * len(imgs)
        preds = run_ocr(imgs, langs, det_m, det_p, rec_m, rec_p)
        pages = []
        for pred in preds:
            lines = [tl.text for tl in pred.text_lines]
            pages.append({'text': '\n'.join(lines)})
        return {'full_text': '\n'.join(pg['text'] for pg in pages),
                'pages': pages, 'n_pages': len(pages)}
    return fn


def make_marker():
    _pip('marker-pdf')
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    converter = PdfConverter(artifact_dict=create_model_dict())
    def fn(orgnr, b):
        rendered = converter(b['pdf'])
        text, _, _ = text_from_rendered(rendered)
        return {'full_text': text}
    return fn


def make_docling():
    _pip('docling')
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    def fn(orgnr, b):
        r = conv.convert(b['pdf'])
        md = r.document.export_to_markdown()
        return {'full_text': md}
    return fn


# === Vision-language and document-understanding engines ===

def make_pix2struct():
    _pip('transformers')
    import torch
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    from PIL import Image
    ckpt = 'google/pix2struct-docvqa-base'
    proc = Pix2StructProcessor.from_pretrained(ckpt)
    model = Pix2StructForConditionalGeneration.from_pretrained(ckpt)
    if torch.cuda.is_available():
        model = model.cuda()
    QS = ['What is the company name?', 'What is the year?', 'What is the årsresultat?',
          'What is the sum eiendeler?', 'What is the sum egenkapital?',
          'What is the driftsresultat?', 'What is the sum kostnader?']
    def fn(orgnr, b):
        n = len(b['page_imgs'])
        targets = b['page_imgs'][1:min(5, n)] if n >= 2 else b['page_imgs']
        out_lines = []
        for img_path in targets:
            img = Image.open(img_path).convert('RGB')
            for q in QS:
                inputs = proc(images=img, return_tensors='pt', text=q)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=48)
                a = proc.decode(out[0], skip_special_tokens=True)
                out_lines.append(f'{q} {a}')
        return {'full_text': '\n'.join(out_lines)}
    return fn


def make_donut():
    _pip('transformers')
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    proc = DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
    model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
    if torch.cuda.is_available():
        model = model.cuda()
    def fn(orgnr, b):
        out_pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            pv = proc(img, return_tensors='pt').pixel_values
            if torch.cuda.is_available():
                pv = pv.cuda()
            decoder_ids = proc.tokenizer('<s_cord-v2>', add_special_tokens=False, return_tensors='pt').input_ids
            if torch.cuda.is_available():
                decoder_ids = decoder_ids.cuda()
            with torch.no_grad():
                out = model.generate(pv, decoder_input_ids=decoder_ids, max_length=512,
                                     bad_words_ids=[[proc.tokenizer.unk_token_id]],
                                     return_dict_in_generate=True)
            txt = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            out_pages.append({'text': txt})
        return {'full_text': '\n'.join(p['text'] for p in out_pages), 'pages': out_pages}
    return fn


def make_udop():
    _pip('transformers')
    import torch
    from transformers import UdopProcessor, UdopForConditionalGeneration
    from PIL import Image
    proc = UdopProcessor.from_pretrained('microsoft/udop-large')
    model = UdopForConditionalGeneration.from_pretrained('microsoft/udop-large')
    if torch.cuda.is_available():
        model = model.cuda()
    def fn(orgnr, b):
        pages = []
        for img_path in b['page_imgs']:
            img = Image.open(img_path).convert('RGB')
            try:
                inputs = proc(img, ['Question answering. Extract all financial numbers and labels from this page.'],
                              return_tensors='pt', text_pair=[''])
                if torch.cuda.is_available():
                    inputs = {k: (v.cuda() if hasattr(v, 'cuda') else v) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=512)
                txt = proc.tokenizer.decode(out[0], skip_special_tokens=True)
                pages.append({'text': txt})
            except Exception as e:
                pages.append({'text': '', 'error': str(e)[:120]})
        return {'full_text': '\n'.join(p.get('text', '') for p in pages), 'pages': pages}
    return fn


def make_layoutlmv3():
    _pip('transformers pytesseract')
    import pytesseract
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            t = pytesseract.image_to_string(p, lang='nor')
            pages.append({'text': t})
        return {'full_text': '\n'.join(pg['text'] for pg in pages), 'pages': pages,
                'note': 'layoutlmv3 has no text-gen head; using tesseract for textual content'}
    return fn


def make_lilt():
    _pip('transformers pytesseract')
    import pytesseract
    def fn(orgnr, b):
        pages = []
        for p in b['page_imgs']:
            t = pytesseract.image_to_string(p, lang='nor')
            pages.append({'text': t})
        return {'full_text': '\n'.join(pg['text'] for pg in pages), 'pages': pages,
                'note': 'LiLT has no decoder head; using tesseract for textual content'}
    return fn


# === Table extractors with text fall-back ===

def make_camelot():
    _apt('ghostscript')
    _pip('camelot-py[cv]')
    import camelot
    def fn(orgnr, b):
        try:
            tables = camelot.read_pdf(b['pdf'], pages='all', flavor='lattice')
        except Exception:
            try:
                tables = camelot.read_pdf(b['pdf'], pages='all', flavor='stream')
            except Exception as e:
                return {'full_text': '', 'error': str(e)[:120]}
        all_txt = []
        for t in tables:
            for _, row in t.df.iterrows():
                all_txt.extend(str(c) for c in row.tolist())
        return {'full_text': '\n'.join(all_txt), 'n_tables': len(tables)}
    return fn


def make_tabula():
    _apt('default-jre')
    _pip('tabula-py')
    import tabula
    def fn(orgnr, b):
        try:
            tables = tabula.read_pdf(b['pdf'], pages='all', stream=True)
        except Exception as e:
            return {'full_text': '', 'error': str(e)[:120]}
        all_txt = []
        for df in tables:
            for _, row in df.iterrows():
                all_txt.extend(str(c) for c in row.tolist())
        return {'full_text': '\n'.join(all_txt), 'n_tables': len(tables)}
    return fn


# === Word-level engines (from ocr-cascade-eval) — preserve token bbox + conf
# and re-cluster integers row-wise. Emit `full_text` (joined per row) so the
# standard score_ocr_text still works AND emit structured `numbers` / `pages`
# for downstream cascade analysis.

import re as _re_word


def _row_cluster_numbers(words, x_gap_threshold=100, row_bin_px=15):
    """Group same-row tokens into integers via median-gap heuristic.

    `words` is a list of dicts with keys: text, bbox=[x0,y0,x1,y1], conf.
    Returns list of {value, n_tokens, avg_conf}.
    """
    by_row = {}
    for w in words:
        cy = (w['bbox'][1] + w['bbox'][3]) / 2
        row_id = int(round(cy / row_bin_px))
        by_row.setdefault(row_id, []).append(w)
    found = []
    for ws in by_row.values():
        ws.sort(key=lambda w: w['bbox'][0])
        i = 0
        while i < len(ws):
            t = ws[i]['text']
            if not _re_word.match(r'^-?\d+$', t):
                i += 1
                continue
            sign = '-' if t.startswith('-') else ''
            digits = _re_word.sub(r'[^\d]', '', t)
            x_anchor = ws[i]['bbox'][2]
            confs = [ws[i]['conf']]
            within_gaps = []
            j = i + 1
            while j < len(ws):
                tn = ws[j]['text']
                if not _re_word.match(r'^\d{1,3}$', tn):
                    break
                gap = ws[j]['bbox'][0] - x_anchor
                if gap >= x_gap_threshold:
                    break
                if within_gaps:
                    median_gap = sorted(within_gaps)[len(within_gaps)//2]
                    if gap > 2.0 * max(median_gap, 15) and gap > 35:
                        break
                within_gaps.append(gap)
                digits += tn
                x_anchor = ws[j]['bbox'][2]
                confs.append(ws[j]['conf'])
                j += 1
            try:
                v = int(sign + digits)
                if abs(v) >= 10:
                    found.append({
                        'value': v,
                        'n_tokens': j - i,
                        'avg_conf': round(sum(confs) / len(confs), 4),
                    })
            except Exception:
                pass
            i = max(j, i + 1)
    return found


def _join_words_to_full_text(per_page_words, row_bin_px=15):
    """Reconstruct page text by row-clustering words, joining within-row by space."""
    out_pages = []
    for words in per_page_words:
        by_row = {}
        for w in words:
            cy = (w['bbox'][1] + w['bbox'][3]) / 2
            row_id = int(round(cy / row_bin_px))
            by_row.setdefault(row_id, []).append(w)
        lines = []
        for row_id in sorted(by_row.keys()):
            ws = sorted(by_row[row_id], key=lambda w: w['bbox'][0])
            lines.append(' '.join(w['text'] for w in ws))
        out_pages.append('\n'.join(lines))
    return '\n\n'.join(out_pages)


def make_tesseract_tsv():
    _apt('tesseract-ocr tesseract-ocr-nor')
    _pip('pytesseract')
    import pytesseract
    def fn(orgnr, b):
        per_page_words = []
        per_page_numbers = []
        for img_path in b['page_imgs']:
            data = pytesseract.image_to_data(
                img_path, lang='nor', output_type=pytesseract.Output.DICT)
            words = []
            for i, w in enumerate(data['text']):
                w = (w or '').strip()
                if not w:
                    continue
                try:
                    conf = float(data['conf'][i])
                except Exception:
                    conf = -1
                if conf < 30:
                    continue
                left = int(data['left'][i])
                top = int(data['top'][i])
                width = int(data['width'][i])
                height = int(data['height'][i])
                words.append({
                    'text': w,
                    'bbox': [left, top, left + width, top + height],
                    'conf': conf,
                })
            per_page_words.append(words)
            per_page_numbers.append(_row_cluster_numbers(words))
        full_text = _join_words_to_full_text(per_page_words)
        all_values = sorted({n['value'] for nums in per_page_numbers for n in nums})
        return {'full_text': full_text,
                'n_pages': len(per_page_words),
                'n_unique_numbers': len(all_values),
                'all_values': all_values,
                'pages': [{'n_words': len(w), 'n_numbers': len(n), 'numbers': n}
                          for w, n in zip(per_page_words, per_page_numbers)]}
    return fn


def make_doctr_bbox():
    _pip('python-doctr[torch]')
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True,
                          det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
    def fn(orgnr, b):
        per_page_words = []
        per_page_numbers = []
        for img_path in b['page_imgs']:
            doc = DocumentFile.from_images(img_path)
            result = model(doc)
            page_obj = result.pages[0]
            h, w_dim = page_obj.dimensions
            words = []
            for block in page_obj.blocks:
                for line in block.lines:
                    for word in line.words:
                        (x0, y0), (x1, y1) = word.geometry
                        words.append({
                            'text': word.value,
                            'conf': round(float(word.confidence), 4),
                            'bbox': [int(x0 * w_dim), int(y0 * h),
                                     int(x1 * w_dim), int(y1 * h)],
                        })
            per_page_words.append(words)
            per_page_numbers.append(_row_cluster_numbers(words))
        full_text = _join_words_to_full_text(per_page_words)
        all_values = sorted({n['value'] for nums in per_page_numbers for n in nums})
        return {'full_text': full_text,
                'n_pages': len(per_page_words),
                'n_unique_numbers': len(all_values),
                'all_values': all_values,
                'pages': [{'n_words': len(w), 'n_numbers': len(n), 'numbers': n,
                           'avg_word_conf': round(sum(x['conf'] for x in w)
                                                  / max(len(w),1), 3)}
                          for w, n in zip(per_page_words, per_page_numbers)]}
    return fn


def make_ocrmypdf_hocr():
    _apt('ocrmypdf tesseract-ocr-nor')
    import os, subprocess, re
    from html.parser import HTMLParser

    class HocrParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.words = []
            self.in_word = False
            self.current_text = ''
            self.current_bbox = None
            self.current_conf = None

        def handle_starttag(self, tag, attrs):
            attrs_d = dict(attrs)
            if tag == 'span' and 'ocrx_word' in (attrs_d.get('class') or ''):
                title = attrs_d.get('title', '')
                bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)
                conf_match = re.search(r'x_wconf\s+(\d+)', title)
                if bbox_match:
                    self.current_bbox = [int(x) for x in bbox_match.groups()]
                if conf_match:
                    self.current_conf = int(conf_match.group(1))
                self.in_word = True
                self.current_text = ''

        def handle_endtag(self, tag):
            if tag == 'span' and self.in_word:
                t = self.current_text.strip()
                if t and self.current_bbox is not None:
                    self.words.append({
                        'text': t,
                        'bbox': self.current_bbox,
                        'conf': self.current_conf or 0,
                    })
                self.in_word = False
                self.current_bbox = None
                self.current_conf = None

        def handle_data(self, data):
            if self.in_word:
                self.current_text += data

    def fn(orgnr, b):
        per_page_words = []
        per_page_numbers = []
        for img_path in b['page_imgs']:
            page_n = os.path.basename(img_path).split('-')[-1].split('.')[0]
            hocr_stem = f'/tmp/_hocr_{orgnr}_{page_n}'
            subprocess.run(['tesseract', img_path, hocr_stem,
                            '-l', 'nor', 'hocr'],
                           capture_output=True, timeout=180)
            hocr_file = hocr_stem + '.hocr'
            if not os.path.exists(hocr_file):
                per_page_words.append([])
                per_page_numbers.append([])
                continue
            parser = HocrParser()
            with open(hocr_file, 'r', encoding='utf-8', errors='replace') as f:
                parser.feed(f.read())
            try:
                os.remove(hocr_file)
            except Exception:
                pass
            per_page_words.append(parser.words)
            per_page_numbers.append(_row_cluster_numbers(parser.words))
        full_text = _join_words_to_full_text(per_page_words)
        all_values = sorted({n['value'] for nums in per_page_numbers for n in nums})
        return {'full_text': full_text,
                'n_pages': len(per_page_words),
                'n_unique_numbers': len(all_values),
                'all_values': all_values,
                'pages': [{'n_words': len(w), 'n_numbers': len(n), 'numbers': n}
                          for w, n in zip(per_page_words, per_page_numbers)]}
    return fn


def make_paddleocr_conf():
    # Use modern paddleocr 3.x for the conf-aware API (predict()/rec_scores/rec_polys).
    # The earlier make_paddleocr() pin (2.7.3) keeps the legacy ocr() API.
    _pip('paddleocr paddlepaddle-gpu')
    import re
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_detection_model_name='PP-OCRv5_mobile_det',
                    text_recognition_model_name='latin_PP-OCRv5_mobile_rec',
                    enable_mkldnn=False, device='cpu')

    def fn(orgnr, b):
        per_page_lines = []
        per_page_numbers = []
        per_page_text = []
        for img_path in b['page_imgs']:
            res = ocr.predict(img_path)
            page = res[0]
            texts = page['rec_texts']
            scores = page['rec_scores']
            polys = page.get('rec_polys') or []
            lines = []
            numbers = []
            for i, (t, s) in enumerate(zip(texts, scores)):
                poly = polys[i] if i < len(polys) else None
                bbox = None
                if poly is not None:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = [int(min(xs)), int(min(ys)),
                            int(max(xs)), int(max(ys))]
                lines.append({'text': t,
                              'conf': round(float(s), 4),
                              'bbox': bbox})
                if re.search(r'\d', t):
                    digits = re.sub(r'[^\d-]', '', t)
                    if digits and re.match(r'^-?\d+$', digits):
                        try:
                            v = int(digits)
                            if abs(v) >= 10:
                                numbers.append({'value': v,
                                                'raw': t,
                                                'conf': round(float(s), 4),
                                                'bbox': bbox,
                                                'low_conf': float(s) < 0.85})
                        except Exception:
                            pass
            per_page_lines.append(lines)
            per_page_numbers.append(numbers)
            per_page_text.append('\n'.join(texts))
        full_text = '\n\n'.join(per_page_text)
        all_values = sorted({n['value'] for nums in per_page_numbers for n in nums})
        return {'full_text': full_text,
                'n_pages': len(per_page_lines),
                'n_unique_numbers': len(all_values),
                'all_values': all_values,
                'pages': [{'n_lines': len(l), 'n_numbers': len(n),
                           'n_low_conf_numbers': sum(1 for x in n if x.get('low_conf')),
                           'numbers': n}
                          for l, n in zip(per_page_lines, per_page_numbers)]}
    return fn


# === Registry ===

ENGINE_FACTORIES = {
    'tesseract':       make_tesseract,
    'easyocr':         make_easyocr,
    'paddleocr':       make_paddleocr,
    'ocrmypdf':        make_ocrmypdf,
    'doctr':           make_doctr,
    'nougat':          make_nougat,
    'trocr':           make_trocr,
    'surya':           make_surya,
    'marker':          make_marker,
    'docling':         make_docling,
    'pix2struct':      make_pix2struct,
    'donut':           make_donut,
    'udop':            make_udop,
    'layoutlmv3':      make_layoutlmv3,
    'lilt':            make_lilt,
    'camelot':         make_camelot,
    'tabula':          make_tabula,
    'tesseract_tsv':   make_tesseract_tsv,
    'doctr_bbox':      make_doctr_bbox,
    'ocrmypdf_hocr':   make_ocrmypdf_hocr,
    'paddleocr_conf':  make_paddleocr_conf,
}

# Heuristic device classification — refined by calibration into engine_calibration.json
DEVICE_HINT = {
    'tesseract':       'cpu',
    'easyocr':         'gpu',
    'paddleocr':       'gpu',
    'ocrmypdf':        'cpu',
    'doctr':           'gpu',
    'nougat':          'gpu',
    'trocr':           'gpu',
    'surya':           'gpu',
    'marker':          'gpu',
    'docling':         'gpu',
    'pix2struct':      'gpu',
    'donut':           'gpu',
    'udop':            'gpu',
    'layoutlmv3':      'cpu',
    'lilt':            'cpu',
    'camelot':         'cpu',
    'tabula':          'cpu',
    'tesseract_tsv':   'cpu',
    'doctr_bbox':      'gpu',
    'ocrmypdf_hocr':   'cpu',
    'paddleocr_conf':  'cpu',
}
