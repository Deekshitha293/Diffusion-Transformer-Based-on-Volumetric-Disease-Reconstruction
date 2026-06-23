import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

try:
    from core.volumetric_stack import VolumetricStacker, IsosurfaceExtractor, VolumeRenderer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.volumetric_stack import VolumetricStacker, IsosurfaceExtractor, VolumeRenderer

st.set_page_config(page_title="Stage 4 — Volumetric Reconstruction", layout="wide", page_icon="🧠")

if not st.session_state.get("preprocessing_done", False):
    st.warning("Please complete Stage 1 first.")
    st.stop()

st.markdown("<h1 style='color: #7F77DD;'>Stage 4 — 3D Volumetric Reconstruction</h1>", unsafe_allow_html=True)
st.markdown("<h3>Coherent assembly of volumetric pathological structures from 2D slices</h3>", unsafe_allow_html=True)
st.divider()

with st.expander("Reconstruction Controls", expanded=True):
    interp_factor = st.slider("Interpolation Factor", min_value=1, max_value=4, value=2, help="Number of interpolated slices between each real slice")
    slice_spacing = st.slider("Slice Spacing (mm)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    iso_threshold = st.slider("Isosurface Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05, help="Voxel intensity threshold for surface extraction")
    smooth_iters = st.slider("Surface Smoothing Iterations", min_value=0, max_value=10, value=3)
    apply_smooth = st.checkbox("Apply Laplacian Surface Smoothing", value=True)
    color_scale = st.radio("Colour Scale", options=["Plasma", "Viridis", "Hot", "Jet"], index=0, horizontal=True)

if st.button("Reconstruct 3D Volume", type="primary"):
    with st.spinner("Assembling 3D volume... computing isosurface..."):
        slices = st.session_state["preprocessed_data"]["slices"]
        
        stacker = VolumetricStacker(slice_spacing=slice_spacing, interpolation_factor=interp_factor)
        volume = stacker.stack_slices(slices)
        stats = stacker.compute_volume_stats()
        
        iso_extractor = IsosurfaceExtractor(iso_value=iso_threshold)
        
        try:
            surface_data = iso_extractor.extract_surface(volume)
            
            if apply_smooth and smooth_iters > 0:
                smoothed_verts = iso_extractor.smooth_surface(surface_data["verts"], surface_data["faces"], iterations=smooth_iters)
                surface_data["verts"] = smoothed_verts
                
            renderer = VolumeRenderer()
            
            iso_fig = renderer.render_isosurface(surface_data, title="3D Lesion Reconstruction", color_scale=color_scale)
            slice_fig = renderer.render_volume_slices(volume)
            
            st.session_state["reconstruction_results"] = stats
            st.session_state["iso_figure"] = iso_fig
            st.session_state["slice_figure"] = slice_fig
            st.session_state["volume_3d"] = volume
            st.session_state["surface_data"] = surface_data
            st.session_state["reconstruction_done"] = True
            
        except ValueError as e:
            st.error(f"Failed to extract surface: {str(e)}. Try lowering the threshold or checking the slices.")

if st.session_state.get("reconstruction_done", False):
    stats = st.session_state["reconstruction_results"]
    iso_fig = st.session_state["iso_figure"]
    slice_fig = st.session_state["slice_figure"]
    volume = st.session_state["volume_3d"]
    
    st.subheader("Volume Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Volume Shape (Z×H×W)", f"{stats['shape'][0]}×{stats['shape'][1]}×{stats['shape'][2]}")
    c2.metric("Total Voxels", f"{stats['voxel_count']:,}")
    c3.metric("Mean Intensity", f"{stats['mean_intensity']:.4f}")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Std Intensity", f"{stats['std_intensity']:.4f}")
    c5.metric("Non-Zero Fraction", f"{stats['non_zero_fraction']*100:.2f}%")
    
    surface_data = st.session_state.get("surface_data", {})
    face_count = surface_data.get("face_count", 0)
    c6.metric("Surface Faces", f"{face_count:,}")
    
    st.subheader("Interactive 3D Lesion Surface")
    st.plotly_chart(iso_fig, use_container_width=True, height=600)
    st.info("Rotate: click-drag | Zoom: scroll | Pan: right-click-drag")
    
    st.subheader("Maximum Intensity Projections")
    st.plotly_chart(slice_fig, use_container_width=True)
    
    st.subheader("Volume Slice Explorer")
    vc1, vc2, vc3 = st.columns(3)
    
    Z, H, W = volume.shape
    
    with vc1:
        ax_idx = st.slider("Axial Slice", min_value=0, max_value=Z-1, value=Z//2)
        fig_ax, ax_ax = plt.subplots(figsize=(3,3))
        ax_ax.imshow(volume[ax_idx, :, :], cmap='gray', vmin=0, vmax=1)
        ax_ax.axis('off')
        fig_ax.patch.set_facecolor('#13152A')
        st.pyplot(fig_ax)
        plt.close(fig_ax)
        
    with vc2:
        cor_idx = st.slider("Coronal Slice", min_value=0, max_value=H-1, value=H//2)
        fig_cor, ax_cor = plt.subplots(figsize=(3,3))
        ax_cor.imshow(volume[:, cor_idx, :], cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax_cor.axis('off')
        fig_cor.patch.set_facecolor('#13152A')
        st.pyplot(fig_cor)
        plt.close(fig_cor)
        
    with vc3:
        sag_idx = st.slider("Sagittal Slice", min_value=0, max_value=W-1, value=W//2)
        fig_sag, ax_sag = plt.subplots(figsize=(3,3))
        ax_sag.imshow(volume[:, :, sag_idx], cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax_sag.axis('off')
        fig_sag.patch.set_facecolor('#13152A')
        st.pyplot(fig_sag)
        plt.close(fig_sag)
