# Faceless YouTube Thumbnail Benchmark Lab

A Streamlit app for descriptive analysis of 14 image and text-in-image features, historical P10/median/P90 reference intervals, image-level bootstrap boundary stability, and rule-based pre-upload thumbnail review.

## Install
1. Install Python 3.10+.
2. Install the **Tesseract OCR executable** separately (not just pytesseract). On Windows, install Tesseract and either add it to PATH or specify its executable path in the app sidebar.
3. In this directory, run:
   ```bash
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

## Workflow
- **Benchmark:** Upload a ZIP containing thumbnail images only. No CSV, IDs, view counts, titles, or other metadata are needed. Images may be nested in folders inside the ZIP. Download feature CSV, reference JSON, reference CSV, bootstrap CSV, and processing log CSV.
- **Assess:** Upload a thumbnail and the saved reference JSON (or use the benchmark from the current session). Inspect feature-level comparisons and download assessment CSV.

## Research cautions
- Historical dataset selection (>50,000 views, faceless video format) must be verified **outside** this image-only app.
- P10–P90 is a **candidate descriptive reference interval**, not an optimum or a performance prediction.
- Bootstrap is **image-level**, not channel-aware; channel dependence cannot be assessed without metadata.
- OCR and Haar face detection are imperfect. The clutter score is an **edge-density proxy**, not a validated universal perceptual clutter measure.
- Numeric outputs depend on the definitions, OpenCV/Tesseract versions and processing settings below. Recompute the historical benchmark if these change.

## Feature definitions (v1)
1. `brightness`: grayscale pixel mean / 255.
2. `contrast`: grayscale pixel standard deviation / 255.
3. `sharpness`: variance of 3x3 Laplacian on grayscale / 1000 (unbounded).
4. `mobile_sharpness`: same Laplacian variance / 1000 after resize to 320x180 (unbounded).
5. `ocr_text_percentage`: union of valid OCR word bounding boxes / image area * 100.
6. `ocr_confidence`: mean valid word confidence / 100; NaN if no valid words.
7. `text_word_count`: number of valid OCR word boxes (integer).
8. `text_median_height`: median valid word-box height / image height; NaN if none.
9. `text_local_contrast`: mean over valid word boxes of abs(mean grayscale in word box - mean grayscale in expanded local ring) / 255; NaN if none.
10. `text_center_x`: area-weighted mean OCR word-box center x / image width; NaN if none.
11. `text_present`: 1 if valid OCR word(s) found, otherwise 0.
12. `face_percentage`: union of Haar-detected face rectangles / image area * 100.
13. `face_count`: number of Haar-detected face rectangles.
14. `clutter_score`: Canny edge pixel fraction, 0–1 (computational proxy).

All features except counts and text presence are continuous; some are unbounded. Text-dependent undefined values are NaN, not zero. OCR requires confidence >= 30 and nonempty word text. Image dimensions are standardized to 1280x720 for analysis, so source aspect ratios are resized (potential distortion); for a rigorous thesis report this decision and its implications. Processing and output are deterministic except for bootstrap random resampling, for which the app records the seed.

## Important
Before presenting the implementation in the thesis, verify that these precise formulas and settings match the code you intend to evaluate. They are documented choices for this updated version, not claims about a previous version of your application.
