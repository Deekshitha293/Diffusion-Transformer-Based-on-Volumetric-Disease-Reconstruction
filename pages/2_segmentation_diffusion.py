import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

try:
    from core.diffusion_mask import DiffusionMaskRefiner, MaskRefiner
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.diffusion_mask import DiffusionMaskRefiner, MaskRefiner

st.set_page_config(page_title="Stage 2 — Segmentation & Diffusion", layout="wide", page_icon="🧠")

if not st.session_state.get("preprocessing_done", False):
    st.warning("Please complete Stage 1 first.")
    st.stop()

st.markdown("<h1 style='color: #7F77DD;'>Stage 2 — Diffusion Mask Refinement</h1>", unsafe_allow_html=True)
st.markdown("<h3>Iterative denoising to extract precise lesion boundaries</h3>", unsafe_allow_html=True)
st.divider()

with st.expander("Diffusion Controls", expanded=True):
    num_steps = st.slider("Number of Diffusion Steps", min_value=10, max_value=100, value=50)
    start_step = st.slider("Diffusion Start Step (t)", min_value=1, max_value=num_steps, value=25)
    open_radius = st.slider("Morphological Open Radius", min_value=1, max_value=7, value=3, step=2)
    close_radius = st.slider("Morphological Close Radius", min_value=1, max_value=11, value=5, step=2)
    
    filenames = st.session_state["preprocessed_data"]["filenames"]
    selected_filename = st.selectbox("Select Slice to Process", options=filenames)

if st.button("Run Diffusion Segmentation", type="primary"):
    with st.spinner("Running diffusion segmentation process..."):
        idx = filenames.index(selected_filename)
        img = st.session_state["preprocessed_data"]["slices"][idx]
        
        refiner = DiffusionMaskRefiner(num_steps=num_steps, beta_end=0.02)
        noisy_initial = refiner.generate_initial_noisy_mask(img, t=start_step)
        reverse_steps = refiner.full_reverse_process(noisy_initial, start_t=start_step)
        
        mask_refiner = MaskRefiner()
        mask_results = mask_refiner.refine_mask(img)
        
        st.session_state["diffusion_results"] = {
            "reverse_steps": reverse_steps,
            "mask_results": mask_results,
            "noisy_initial": noisy_initial,
            "original_img": img,
            "start_step": start_step
        }
        st.session_state["diffusion_done"] = True

if st.session_state.get("diffusion_done", False):
    res = st.session_state["diffusion_results"]
    
    st.subheader("Diffusion Process Visualisation")
    
    cols = st.columns(5)
    steps_list = res["reverse_steps"]
    
    # Selection of 3 evenly spaced reverse steps
    if len(steps_list) >= 3:
        selected_indices = np.linspace(0, len(steps_list)-1, 3, dtype=int)
    else:
        # Fallback if list is too small
        selected_indices = [0] * 3
        
    titles = [
        "Original Slice",
        f"Noisy (t={res['start_step']})",
        f"Denoising Step 1",
        f"Denoising Step 2",
        f"Denoising Final"
    ]
    
    images_to_show = [
        res["original_img"],
        res["noisy_initial"],
        steps_list[selected_indices[0]],
        steps_list[selected_indices[1]],
        steps_list[selected_indices[2]]
    ]
    
    for c, title, img_show in zip(cols, titles, images_to_show):
        with c:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.imshow(img_show, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(title, fontsize=10, color='white')
            # Theme it cleanly
            fig.patch.set_facecolor('#13152A')
            st.pyplot(fig)
            plt.close(fig)
            
    st.subheader("Mask Refinement Results")
    m_cols = st.columns(3)
    
    m_titles = ["Original Slice", "Raw Otsu Mask", "Refined Morphological Mask"]
    m_images = [
        res["original_img"],
        res["mask_results"]["raw_mask"],
        res["mask_results"]["refined_mask"]
    ]
    m_cmaps = ['gray', 'hot', 'hot']
    
    for i, c in enumerate(m_cols):
        with c:
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(m_images[i], cmap=m_cmaps[i])
            ax.axis('off')
            ax.set_title(m_titles[i], color='white')
            fig.patch.set_facecolor('#13152A')
            st.pyplot(fig)
            plt.close(fig)
            
    st.subheader("Detected Lesion Regions")
    st.metric("Lesions Detected", res["mask_results"]["lesion_count"])
    
    regions = res["mask_results"]["regions"]
    if len(regions) > 0:
        import pandas as pd
        df = pd.DataFrame(regions)
        df["Centroid Y"] = df["centroid"].apply(lambda x: x[0])
        df["Centroid X"] = df["centroid"].apply(lambda x: x[1])
        df = df[["area", "Centroid Y", "Centroid X", "eccentricity", "solidity"]]
        df.columns = ["Area", "Centroid Y", "Centroid X", "Eccentricity", "Solidity"]
        
        df["Centroid Y"] = df["Centroid Y"].round(3)
        df["Centroid X"] = df["Centroid X"].round(3)
        df["Eccentricity"] = df["Eccentricity"].round(3)
        df["Solidity"] = df["Solidity"].round(3)
        
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No lesion regions with area > 100 pixels were detected.")
