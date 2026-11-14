"""Descriptive image-only thumbnail benchmark engine (v1)."""
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output

FEATURES = [
    'brightness', 'contrast', 'sharpness', 'mobile_sharpness',
    'ocr_text_percentage', 'ocr_confidence', 'text_word_count',
    'text_median_height', 'text_local_contrast', 'text_center_x',
    'text_present', 'face_percentage', 'face_count', 'clutter_score'
]
BINARY = {'text_present'}
DISCRETE = {'text_word_count', 'face_count'}
NUMERIC = [x for x in FEATURES if x not in BINARY]
VERSION = 'thumbnail-features-v1'
CONFIG = {'width': 1280, 'height': 720, 'mobile_width': 320,
          'mobile_height': 180, 'ocr_min_confidence': 30,
          'ocr_psm': 11, 'canny_thresholds': [100, 200],
          'face_detector': 'opencv_haar_frontalface_default',
          'face_scale_factor': 1.1, 'face_min_neighbors': 5,
          'face_min_size': [20, 20]}
ALLOWED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024


def _decode(data):
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError('Image cannot be decoded')
    return cv2.resize(image, (CONFIG['width'], CONFIG['height']), interpolation=cv2.INTER_AREA)


def _union_area(rects, height, width):
    if not rects:
        return 0
    mask = np.zeros((height, width), np.uint8)
    for x, y, w, h in rects:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return int(mask.sum())


def _local_contrast(gray, rect):
    x, y, w, h = rect
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    margin = max(3, int(round(max(w, h) * 0.2)))
    ex0, ey0 = max(0, x0-margin), max(0, y0-margin)
    ex1, ey1 = min(W, x1+margin), min(H, y1+margin)
    patch = gray[ey0:ey1, ex0:ex1]
    ring = np.ones(patch.shape, dtype=bool)
    ring[y0-ey0:y1-ey0, x0-ex0:x1-ex0] = False
    if not ring.any():
        return np.nan
    return abs(float(gray[y0:y1, x0:x1].mean()) - float(patch[ring].mean())) / 255.0


def extract_features(data):
    img = _decode(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    mobile = cv2.resize(gray, (CONFIG['mobile_width'], CONFIG['mobile_height']), interpolation=cv2.INTER_AREA)
    out = {
        'brightness': float(gray.mean() / 255.0),
        'contrast': float(gray.std() / 255.0),
        'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var() / 1000.0),
        'mobile_sharpness': float(cv2.Laplacian(mobile, cv2.CV_64F, ksize=3).var() / 1000.0),
    }
    # OCR data is extracted at the same standardized image size.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d = pytesseract.image_to_data(rgb, config=f"--psm {CONFIG['ocr_psm']}", output_type=Output.DICT)
    words = []
    for i, raw in enumerate(d['text']):
        word = str(raw).strip()
        try:
            conf = float(d['conf'][i])
        except (TypeError, ValueError):
            continue
        if not word or not np.isfinite(conf) or conf < CONFIG['ocr_min_confidence']:
            continue
        rect = (int(d['left'][i]), int(d['top'][i]), int(d['width'][i]), int(d['height'][i]))
        if rect[2] > 0 and rect[3] > 0:
            words.append((rect, conf))
    rects = [r for r, _ in words]
    out['ocr_text_percentage'] = 100.0 * _union_area(rects, H, W) / (H * W)
    out['ocr_confidence'] = float(np.mean([c for _, c in words]) / 100.0) if words else np.nan
    out['text_word_count'] = len(words)
    out['text_median_height'] = float(np.median([r[3] / H for r in rects])) if rects else np.nan
    local_values = [_local_contrast(gray, r) for r in rects]
    local_values = [v for v in local_values if np.isfinite(v)]
    out['text_local_contrast'] = float(np.mean(local_values)) if local_values else np.nan
    areas = [r[2] * r[3] for r in rects]
    out['text_center_x'] = float(np.average([(r[0] + r[2] / 2) / W for r in rects], weights=areas)) if rects else np.nan
    out['text_present'] = int(bool(words))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise RuntimeError('OpenCV Haar face cascade unavailable')
    faces = detector.detectMultiScale(gray, scaleFactor=CONFIG['face_scale_factor'],
                                     minNeighbors=CONFIG['face_min_neighbors'],
                                     minSize=tuple(CONFIG['face_min_size']))
    face_rects = [tuple(map(int, r)) for r in faces]
    out['face_percentage'] = 100.0 * _union_area(face_rects, H, W) / (H * W)
    out['face_count'] = len(face_rects)
    edges = cv2.Canny(gray, *CONFIG['canny_thresholds'])
    out['clutter_score'] = float(np.count_nonzero(edges) / (H * W))
    return out


def process_zip(blob, max_images=3000):
    rows, logs = [], []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        total_uncompressed = 0
        seen = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            filename = info.filename
            if PurePosixPath(filename).suffix.lower() not in ALLOWED:
                continue
            seen += 1
            if seen > max_images:
                raise ValueError(f'Archive has more than {max_images} supported images')
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_TOTAL_BYTES:
                raise ValueError('Archive exceeds total uncompressed-size limit (500 MB)')
            if info.file_size > MAX_FILE_BYTES:
                logs.append({'file': filename, 'status': 'skipped', 'detail': 'Image exceeds 15 MB'})
                continue
            try:
                with archive.open(info) as f:
                    record = extract_features(f.read())
                rows.append({'file': filename, **record})
                logs.append({'file': filename, 'status': 'processed', 'detail': ''})
            except Exception as e:
                logs.append({'file': filename, 'status': 'failed', 'detail': str(e)[:200]})
    return pd.DataFrame(rows, columns=['file'] + FEATURES), pd.DataFrame(logs, columns=['file', 'status', 'detail'])


def _clean_float(v):
    return float(v) if pd.notna(v) and np.isfinite(v) else None


def build_reference(df, bootstrap_iterations=500, seed=42):
    if df.empty:
        raise ValueError('No valid image measurements available')
    rng = np.random.default_rng(int(seed))
    stats, stability = [], []
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors='coerce').dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        if feature in BINARY:
            stats.append({'feature': feature, 'type': 'binary', 'valid_n': n,
                          'p10': None, 'median': None, 'p90': None,
                          'present_count': int(np.count_nonzero(values == 1)),
                          'present_rate': _clean_float(np.mean(values == 1)) if n else None})
            continue
        if not n:
            stats.append({'feature': feature, 'type': 'discrete' if feature in DISCRETE else 'continuous',
                          'valid_n': 0, 'p10': None, 'median': None, 'p90': None,
                          'present_count': None, 'present_rate': None})
            continue
        p10, median, p90 = np.percentile(values, [10, 50, 90], method='linear')
        stats.append({'feature': feature, 'type': 'discrete' if feature in DISCRETE else 'continuous',
                      'valid_n': n, 'p10': float(p10), 'median': float(median), 'p90': float(p90),
                      'present_count': None, 'present_rate': None})
        if bootstrap_iterations >= 2:
            bounds = np.empty((bootstrap_iterations, 2), dtype=float)
            for b in range(bootstrap_iterations):
                sample = rng.choice(values, size=n, replace=True)
                bounds[b] = np.percentile(sample, [10, 90], method='linear')
            for j, boundary in enumerate(['p10', 'p90']):
                ci_lo, ci_hi = np.percentile(bounds[:, j], [2.5, 97.5], method='linear')
                stability.append({'feature': feature, 'boundary': boundary, 'valid_n': n,
                                  'bootstrap_iterations': bootstrap_iterations,
                                  'bootstrap_se': float(np.std(bounds[:, j], ddof=1)),
                                  'bootstrap_ci_2_5': float(ci_lo), 'bootstrap_ci_97_5': float(ci_hi)})
    reference = {'version': VERSION, 'config': CONFIG, 'sample_size_processed': int(len(df)),
                 'bootstrap_iterations': int(bootstrap_iterations), 'bootstrap_seed': int(seed),
                 'selection_criterion': 'Historical images externally selected from faceless YouTube videos with strictly more than 50000 views; not verified by this app',
                 'interpretation': 'Descriptive candidate P10-P90 reference intervals; not optimal ranges or performance predictions',
                 'features': {row['feature']: {k: v for k, v in row.items() if k != 'feature'} for row in stats}}
    return reference, pd.DataFrame(stats), pd.DataFrame(stability, columns=['feature', 'boundary', 'valid_n', 'bootstrap_iterations', 'bootstrap_se', 'bootstrap_ci_2_5', 'bootstrap_ci_97_5'])


def assess(features, reference):
    if reference.get('version') != VERSION or reference.get('config') != CONFIG:
        raise ValueError('Reference version/configuration does not match this feature extractor. Rebuild the reference with this version.')
    rows = []
    for name in FEATURES:
        val = features.get(name)
        ref = reference['features'].get(name)
        if ref is None:
            status, note = 'unavailable', 'Reference value is missing.'
            lo = med = hi = None
        elif name in BINARY:
            lo = med = hi = None
            rate = ref.get('present_rate')
            status = 'text detected' if val == 1 else 'no text detected'
            note = f'Historical detected-text frequency: {rate:.1%}.' if rate is not None else 'Historical frequency unavailable.'
        else:
            lo, med, hi = ref.get('p10'), ref.get('median'), ref.get('p90')
            if val is None or not np.isfinite(val) or lo is None or hi is None:
                status, note = 'unavailable', 'Measurement or historical reference is unavailable.'
            elif val < lo:
                status, note = 'below reference', 'Below the historical P10; review only if this characteristic matters for your design.'
            elif val > hi:
                status, note = 'above reference', 'Above the historical P90; review only if this characteristic matters for your design.'
            else:
                status, note = 'within reference', 'Within the historical P10–P90 candidate reference interval.'
        rows.append({'feature': name, 'value': val, 'p10': lo, 'median': med, 'p90': hi,
                     'assessment': status, 'feedback': note})
    return pd.DataFrame(rows)


def reference_json(reference):
    return json.dumps(reference, indent=2, allow_nan=False)
