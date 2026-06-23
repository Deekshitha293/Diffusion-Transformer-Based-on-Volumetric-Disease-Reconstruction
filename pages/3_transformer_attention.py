import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

try:
    from core.transformer_attention import InterSliceAttention, get_patch_attention_map
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.transformer_attention import InterSliceAttention, get_patch_attention_map

st.set_page_config(page_title="Stage 3 — Transformer Attention", layout="wide", page_icon="🧠")

if not st.session_state.get("preprocessing_done", False):
    st.warning("Please complete Stage 1 first.")
    st.stop()

st.markdown("<h1 style='color: #7F77DD;'>Stage 3 — Transformer Global Attention</h1>", unsafe_allow_html=True)
st.markdown("<h3>Vision Transformer captures inter-slice spatial dependencies</h3>", unsafe_allow_html=True)
st.divider()

with st.expander("Attention Controls", expanded=True):
    num_heads = st.slider("Number of Attention Heads", min_value=1, max_value=16, value=8)
    embed_dim = st.slider("Embedding Dimension", min_value=64, max_value=512, value=256, step=64)
    
    total_slice_count = st.session_state["preprocessed_data"]["count"]
    num_slices = st.slider("Number of Slices to Analyse", min_value=3, max_value=20, value=min(10, total_slice_count))
    
    filenames = st.session_state["preprocessed_data"]["filenames"]
    selected_filename = st.selectbox("Reference Slice for Patch Attention", options=filenames)

if st.button("Compute Transformer Attention", type="primary"):
    with st.spinner("Computing transformer attention maps..."):
        slices = st.session_state["preprocessed_data"]["slices"][:num_slices]
        
        with torch.no_grad():
            attn_model = InterSliceAttention(embed_dim=embed_dim, num_heads=num_heads)
            inter_slice_results = attn_model.compute_inter_slice_attention(slices)
            
            idx = filenames.index(selected_filename)
            ref_img = st.session_state["preprocessed_data"]["slices"][idx]
            
            patch_attn_map = get_patch_attention_map(ref_img, embed_dim=embed_dim, num_heads=num_heads)
        
        st.session_state["attention_results"] = {
            "inter_slice": inter_slice_results,
            "patch_attn_map": patch_attn_map,
            "ref_img": ref_img,
        }
        st.session_state["attention_done"] = True

if st.session_state.get("attention_done", False):
    res = st.session_state["attention_results"]
    attn_matrix = res["inter_slice"]["attention_matrix"]
    patch_attn = res["patch_attn_map"]
    ref_img = res["ref_img"]
    
    st.subheader("Inter-Slice Attention Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(attn_matrix, cmap='viridis')
    cbar = fig.colorbar(cax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    num_slices_analyzed = res["inter_slice"]["num_slices"]
    ax.set_xticks(range(num_slices_analyzed))
    ax.set_yticks(range(num_slices_analyzed))
    
    ax.set_xlabel("Slice Index", color='white')
    ax.set_ylabel("Slice Index", color='white')
    ax.set_title("Inter-Slice Attention Weights", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#13152A')
    ax.set_facecolor('#0D0E1A')
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.subheader("Per-Slice Attention Heatmap")
    c1, c2 = st.columns(2)
    
    ref_img_uint8 = (np.clip(ref_img, 0, 1) * 255).astype(np.uint8)
    
    patch_attn_uint8 = (np.clip(patch_attn, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(patch_attn_uint8, cv2.COLORMAP_JET)
    
    ref_img_3ch = cv2.cvtColor(ref_img_uint8, cv2.COLOR_GRAY2BGR)
    # Applying overlay correctly (convert to RGB for Streamlit image rendering)
    overlaid = cv2.addWeighted(ref_img_3ch, 0.6, heatmap_colored, 0.4, 0)
    overlaid_rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
    
    with c1:
        st.image(ref_img_uint8, caption="Reference Slice Grayscale", use_column_width=True)
    with c2:
        st.image(overlaid_rgb, caption="Attention Heatmap Overlaid", use_column_width=True)
        
    st.subheader("Attention Statistics")
    m1, m2, m3, m4 = st.columns(4)
    
    max_attn = np.max(attn_matrix)
    min_attn = np.min(attn_matrix)
    mean_attn = np.mean(attn_matrix)
    
    row_sums = attn_matrix.sum(axis=1, keepdims=True)
    p = attn_matrix / (row_sums + 1e-8)
    entropy = -np.sum(p * np.log(p + 1e-8))
    
    m1.metric("Max Attention", f"{max_attn:.4f}")
    m2.metric("Min Attention", f"{min_attn:.4f}")
    m3.metric("Mean Attention", f"{mean_attn:.4f}")
    m4.metric("Attention Entropy", f"{entropy:.4f}")
