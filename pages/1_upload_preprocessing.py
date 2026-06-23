import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import cv2

# Ensure core can be imported
try:
    from core.preprocessing import SlicePreprocessor, FAISSSliceIndex, generate_sample_slices
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.preprocessing import SlicePreprocessor, FAISSSliceIndex, generate_sample_slices

st.set_page_config(page_title="Stage 1 — Upload & Preprocessing", layout="wide", page_icon="🧠")

st.markdown("<h1 style='color: #7F77DD;'>Stage 1 — Upload & Preprocessing</h1>", unsafe_allow_html=True)
st.markdown("<h3>Prepare and index your volumetric imaging data</h3>", unsafe_allow_html=True)
st.divider()

st.subheader("Upload Medical Image Slices")
st.info("Supported formats: PNG, JPG, JPEG, NIfTI (.nii, .nii.gz). Upload 10–300 slices representing a single volumetric scan.")

file_uploader = st.file_uploader("Upload scan slices", type=["png", "jpg", "jpeg", "nii", "nii.gz"], accept_multiple_files=True, key="slice_upload")

if st.button("Use Sample Data (20 Synthetic Slices)"):
    st.session_state["sample_mode"] = True

with st.expander("Preprocessing Configuration", expanded=True):
    target_size_val = st.slider("Target Image Size", min_value=64, max_value=512, value=256, step=64)
    st.caption("All slices resized to NxN pixels")
    
    noise_sigma = st.slider("Gaussian Noise Sigma", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    st.caption("Smoothing strength for denoising")
    
    faiss_top_k = st.slider("FAISS Top-K Retrieval", min_value=1, max_value=20, value=5, step=1)
    st.caption("Number of similar slices to retrieve")
    
    do_align = st.checkbox("Enable Slice Alignment", value=True)
    show_embed_viz = st.checkbox("Show Embedding Visualisation", value=False)

if st.button("Run Preprocessing Pipeline", type="primary"):
    with st.spinner("Preprocessing slices... this may take a moment"):
        preprocessor = SlicePreprocessor(target_size=(target_size_val, target_size_val), noise_sigma=noise_sigma)
        
        file_list = []
        if st.session_state.get("sample_mode", False):
            sample_slices = generate_sample_slices(20)
            for i, s in enumerate(sample_slices):
                s_uint8 = (s * 255).astype(np.uint8)
                success, buffer = cv2.imencode('.png', s_uint8)
                if success:
                    file_list.append((f"sample_slice_{i:03d}.png", buffer.tobytes()))
        else:
            if not file_uploader:
                st.warning("Please upload files or use sample data.")
                st.stop()
            file_list = [(f.name, f.getvalue()) for f in file_uploader]
            
        if file_list:
            processed_data = preprocessor.process_batch(file_list, do_align=do_align)
            
            faiss_index = FAISSSliceIndex(128)
            faiss_index.build_index(processed_data["embeddings"], processed_data["filenames"])
            
            st.session_state["preprocessed_data"] = processed_data
            st.session_state["faiss_index"] = faiss_index
            st.session_state["preprocessing_done"] = True
            st.success("Preprocessing complete! Proceed to Stage 2.")

if st.session_state.get("preprocessing_done", False):
    processed_data = st.session_state["preprocessed_data"]
    faiss_index = st.session_state["faiss_index"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Slices Loaded", str(processed_data["count"]))
    col2.metric("Image Shape", f"{processed_data['shape'][0]} × {processed_data['shape'][1]}")
    col3.metric("Embedding Dim", "128")
    col4.metric("FAISS Index Size", str(faiss_index.get_index_stats()["total_slices"]))
    
    st.subheader("Slice Preview Grid")
    preview_slices = processed_data["slices"][:12]
    preview_names = processed_data["filenames"][:12]
    
    cols = st.columns(4)
    for i, (slice_arr, name) in enumerate(zip(preview_slices, preview_names)):
        with cols[i % 4]:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(slice_arr, cmap='gray')
            ax.axis('off')
            ax.set_title(name[:15], fontsize=8, color='white')
            # Theme it cleanly
            fig.patch.set_facecolor('#13152A')
            st.pyplot(fig)
            plt.close(fig)
            
    st.subheader("FAISS Similarity Search Demo")
    query_idx = st.slider("Query slice index", min_value=0, max_value=processed_data["count"]-1, value=0)
    
    query_emb = processed_data["embeddings"][query_idx]
    results = faiss_index.query(query_emb, top_k=faiss_top_k)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df[["rank", "filename", "distance"]]
        df.columns = ["Rank", "Filename", "Distance"]
        st.dataframe(df, use_container_width=True)
        
    if show_embed_viz:
        pca = PCA(n_components=2)
        if processed_data["count"] >= 2:
            reduced_embs = pca.fit_transform(processed_data["embeddings"])
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            fig2.patch.set_facecolor('#13152A')
            ax2.set_facecolor('#0D0E1A')
            ax2.tick_params(colors='white')
            
            scatter = ax2.scatter(reduced_embs[:, 0], reduced_embs[:, 1], c=range(processed_data["count"]), cmap='viridis', edgecolors='white', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("Slice Index", color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            ax2.set_title("PCA of Slice Embeddings", color='white')
            
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.warning("Not enough slices to perform PCA.")
