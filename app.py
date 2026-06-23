import streamlit as st
import config

st.set_page_config(
    page_title="SegDT",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("assets/style.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"<h2>🧠 {config.APP_TITLE}</h2>", unsafe_allow_html=True)
    st.divider()
    st.write("Select a pipeline stage from the pages above")
    
    st.markdown("### System Status")
    st.success("FAISS Index: Ready")
    st.success("Diffusion Engine: Loaded")
    st.success("Transformer: Active")
    
    st.markdown("---")
    st.markdown("<small>SegDT Research Pipeline v1.0<br>Medical AI · Radiology</small>", unsafe_allow_html=True)

# Main area
st.markdown(f"<h1 style='text-align: center; color: {config.THEME_COLOR_PRIMARY}; font-size: 56px;'>SegDT</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center; color: #B4B2A9;'>{config.APP_SUBTITLE}</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; color: #B4B2A9;'>Seeing disease as structure, not slices.</p>", unsafe_allow_html=True)
st.divider()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pipeline Stages", "5", "End-to-End")
col2.metric("Max Slices", str(config.MAX_SLICES), "per volume")
col3.metric("Diffusion Steps", str(config.DEFAULT_DIFFUSION_STEPS), "default")
col4.metric("Transformer Heads", str(config.TRANSFORMER_HEADS), "attention")

# Info cards row
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
    **The Challenge**
    - Fragmented slice-by-slice diagnosis
    - Structural ambiguity at lesion boundaries
    - Local CNN reasoning limits global context
    """)

with c2:
    st.info("""
    **Our Solution**
    - Diffusion-based mask refinement
    - Transformer global attention across slices
    - Coherent 3D volumetric stacking
    """)

with c3:
    st.info("""
    **Clinical Impact**
    - Reduced radiologist cognitive load
    - Enhanced volumetric visualisation
    - Precise disease quantification
    """)

# Banner
st.markdown("<br>", unsafe_allow_html=True)
st.info("Navigate through the 5 pipeline stages using the sidebar. Begin with Stage 1: Upload & Preprocessing.")

# Pipeline flow diagram
st.markdown("<br>", unsafe_allow_html=True)
flow_html = f"""
<div style="display: flex; justify-content: space-between; align-items: center; background-color: {config.THEME_COLOR_BG}; padding: 20px; border-radius: 8px;">
    <div style="background-color: {config.THEME_COLOR_SURFACE}; border: 1px solid {config.THEME_COLOR_PRIMARY}; padding: 15px; border-radius: 8px; color: white; text-align: center; flex: 1;">Upload & Preprocess</div>
    <div style="color: {config.THEME_COLOR_PRIMARY}; padding: 0 10px; font-weight: bold;">→</div>
    <div style="background-color: {config.THEME_COLOR_SURFACE}; border: 1px solid {config.THEME_COLOR_PRIMARY}; padding: 15px; border-radius: 8px; color: white; text-align: center; flex: 1;">Diffusion Segmentation</div>
    <div style="color: {config.THEME_COLOR_PRIMARY}; padding: 0 10px; font-weight: bold;">→</div>
    <div style="background-color: {config.THEME_COLOR_SURFACE}; border: 1px solid {config.THEME_COLOR_PRIMARY}; padding: 15px; border-radius: 8px; color: white; text-align: center; flex: 1;">Transformer Attention</div>
    <div style="color: {config.THEME_COLOR_PRIMARY}; padding: 0 10px; font-weight: bold;">→</div>
    <div style="background-color: {config.THEME_COLOR_SURFACE}; border: 1px solid {config.THEME_COLOR_PRIMARY}; padding: 15px; border-radius: 8px; color: white; text-align: center; flex: 1;">3D Reconstruction</div>
    <div style="color: {config.THEME_COLOR_PRIMARY}; padding: 0 10px; font-weight: bold;">→</div>
    <div style="background-color: {config.THEME_COLOR_SURFACE}; border: 1px solid {config.THEME_COLOR_PRIMARY}; padding: 15px; border-radius: 8px; color: white; text-align: center; flex: 1;">Quantitative Analysis</div>
</div>
"""
st.markdown(flow_html, unsafe_allow_html=True)
