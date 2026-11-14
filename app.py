import io
import json

import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

from engine import FEATURES, CONFIG, assess, build_reference, extract_features, process_zip, reference_json

st.set_page_config(page_title='Thumbnail Benchmark Lab', layout='wide')
st.title('Faceless YouTube Thumbnail Benchmark Lab')
st.caption('')

with st.sidebar:
    st.header('Configuration')
    tesseract_path = st.text_input('Tesseract executable path (optional)', placeholder=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    if tesseract_path.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_path.strip()
    st.info('Requires the Tesseract OCR program to be installed separately.')
    st.caption('Fixed analysis resolution: 1280 × 720. Mobile sharpness: 320 × 180. OCR confidence threshold: 30.')
    st.warning('P10–P90 ranges are descriptive candidate references, not proven optimal design settings.')

benchmark_tab, assess_tab, methodology_tab = st.tabs(['1. Build benchmark', '2. Assess new thumbnail', '3. Methods and limitations'])

with benchmark_tab:
    st.subheader('Historical thumbnail ZIP')
    st.write('Upload image files only. The app cannot check historical views, faceless eligibility, channel counts, or selection bias.')
    archive = st.file_uploader('ZIP containing historical thumbnails', type=['zip'], key='archive')
    col1, col2 = st.columns(2)
    with col1:
        iterations = st.number_input('Image-level bootstrap iterations', min_value=0, max_value=5000, value=500, step=100)
    with col2:
        seed = st.number_input('Bootstrap random seed', min_value=0, max_value=2147483647, value=42, step=1)
    if st.button('Process images and build reference', type='primary', disabled=archive is None):
        try:
            with st.spinner('Processing images. OCR and face detection may take several minutes for large archives...'):
                features_df, logs_df = process_zip(archive.getvalue())
                reference, stats_df, bootstrap_df = build_reference(features_df, int(iterations), int(seed))
            st.session_state['reference'] = reference
            st.session_state['features_df'] = features_df
            st.session_state['logs_df'] = logs_df
            st.session_state['stats_df'] = stats_df
            st.session_state['bootstrap_df'] = bootstrap_df
            st.success(f'Processed {len(features_df)} images; {len(logs_df) - len(features_df)} skipped/failed supported files.')
        except Exception as exc:
            st.error(f'Benchmark could not be generated: {exc}')
    if 'reference' in st.session_state:
        features_df = st.session_state['features_df']
        stats_df = st.session_state['stats_df']
        bootstrap_df = st.session_state['bootstrap_df']
        logs_df = st.session_state['logs_df']
        st.subheader('P10 / median / P90 and text-presence frequency')
        st.dataframe(stats_df, use_container_width=True)
        st.subheader('Feature measurements')
        st.dataframe(features_df.head(100), use_container_width=True)
        st.subheader('Bootstrap boundary stability')
        if bootstrap_df.empty:
            st.info('Bootstrap disabled or insufficient iterations.')
        else:
            st.dataframe(bootstrap_df, use_container_width=True)
        st.subheader('Download outputs')
        st.download_button('Feature dataset CSV', features_df.to_csv(index=False).encode(), 'thumbnail_features.csv', 'text/csv')
        st.download_button('Reference summary CSV', stats_df.to_csv(index=False).encode(), 'reference_summary.csv', 'text/csv')
        st.download_button('Reference JSON', reference_json(st.session_state['reference']).encode(), 'thumbnail_reference.json', 'application/json')
        st.download_button('Bootstrap stability CSV', bootstrap_df.to_csv(index=False).encode(), 'bootstrap_stability.csv', 'text/csv')
        st.download_button('Processing log CSV', logs_df.to_csv(index=False).encode(), 'processing_log.csv', 'text/csv')

with assess_tab:
    st.subheader('Pre-upload assessment')
    uploaded_ref = st.file_uploader('Optional: load a previously exported reference JSON', type=['json'])
    active_ref = st.session_state.get('reference')
    if uploaded_ref is not None:
        try:
            active_ref = json.loads(uploaded_ref.getvalue().decode('utf-8'))
        except Exception as exc:
            st.error(f'Invalid reference JSON: {exc}')
            active_ref = None
    image_file = st.file_uploader('Upload a proposed thumbnail', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
    if image_file is not None:
        try:
            st.image(Image.open(io.BytesIO(image_file.getvalue())), caption='Proposed thumbnail', width=500)
        except Exception:
            st.warning('Image preview unavailable.')
    if st.button('Analyze and assess thumbnail', disabled=image_file is None or active_ref is None):
        try:
            with st.spinner('Extracting the 14 features...'):
                extracted = extract_features(image_file.getvalue())
                result = assess(extracted, active_ref)
            st.session_state['last_assessment'] = result
            st.success('Assessment complete. Results are descriptive comparisons, not performance predictions.')
        except Exception as exc:
            st.error(f'Assessment failed: {exc}')
    if active_ref is None:
        st.info('Build a benchmark in tab 1 or upload a reference JSON.')
    if 'last_assessment' in st.session_state:
        st.dataframe(st.session_state['last_assessment'], use_container_width=True)
        st.download_button('Assessment CSV', st.session_state['last_assessment'].to_csv(index=False).encode(), 'thumbnail_assessment.csv', 'text/csv')

with methodology_tab:
    st.markdown('''### Interpretation
- Historical images must be selected externally using the **strictly more than 50,000 views** criterion.
- Faceless describes the video presentation style, **not** absence of faces in thumbnails.
- OCR missing values are preserved as unavailable, not converted to zero.
- Text presence is reported as a binary frequency. Face count and word count are discrete.
- Bootstrap resamples **images**, not channels; results cannot account for within-channel dependence.
- No causal, predictive, or CTR claim is supported by these benchmarks.

### Feature formulas
See the bundled README for the exact 14 definitions. All images are resized to **1280 × 720** for consistent extraction; this can distort source images with different aspect ratios. OCR uses Tesseract and face detection uses OpenCV Haar cascade. The clutter score is an edge-density proxy. The methods and parameters should be documented in the thesis and evaluated against actual outputs.
''')
    st.json(CONFIG)
