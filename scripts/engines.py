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
    _pip('paddleocr paddlepaddle-gpu')
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
    _pip('nougat-ocr')
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
    _pip('surya-ocr')
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


# === Registry ===

ENGINE_FACTORIES = {
    'tesseract':    make_tesseract,
    'easyocr':      make_easyocr,
    'paddleocr':    make_paddleocr,
    'ocrmypdf':     make_ocrmypdf,
    'doctr':        make_doctr,
    'nougat':       make_nougat,
    'trocr':        make_trocr,
    'surya':        make_surya,
    'marker':       make_marker,
    'docling':      make_docling,
    'pix2struct':   make_pix2struct,
    'donut':        make_donut,
    'udop':         make_udop,
    'layoutlmv3':   make_layoutlmv3,
    'lilt':         make_lilt,
    'camelot':      make_camelot,
    'tabula':       make_tabula,
}

# Heuristic device classification — refined by calibration into engine_calibration.json
DEVICE_HINT = {
    'tesseract':    'cpu',
    'easyocr':      'gpu',
    'paddleocr':    'gpu',
    'ocrmypdf':     'cpu',
    'doctr':        'gpu',
    'nougat':       'gpu',
    'trocr':        'gpu',
    'surya':        'gpu',
    'marker':       'gpu',
    'docling':      'gpu',
    'pix2struct':   'gpu',
    'donut':        'gpu',
    'udop':         'gpu',
    'layoutlmv3':   'cpu',
    'lilt':         'cpu',
    'camelot':      'cpu',
    'tabula':       'cpu',
}
