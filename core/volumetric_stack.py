import numpy as np
import scipy.ndimage
import scipy.interpolate
import skimage.measure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2

class VolumetricStacker:
    def __init__(self, slice_spacing=1.0, interpolation_factor=2):
        self.slice_spacing = slice_spacing
        self.interpolation_factor = interpolation_factor
        self.volume = None

    def stack_slices(self, slices: list) -> np.ndarray:
        volume = np.stack(slices, axis=0) # shape (N, 256, 256)
        if self.interpolation_factor > 1:
            volume = scipy.ndimage.zoom(volume, zoom=(self.interpolation_factor, 1, 1), order=1)
        volume = scipy.ndimage.gaussian_filter(volume, sigma=0.8)
        self.volume = volume
        return self.volume

    def get_projection(self, axis='axial') -> np.ndarray:
        if self.volume is None: return None
        if axis == 'axial':
            mip = np.max(self.volume, axis=0)
        elif axis == 'coronal':
            mip = np.max(self.volume, axis=1)
        elif axis == 'sagittal':
            mip = np.max(self.volume, axis=2)
        else:
            raise ValueError("Unknown axis")
            
        m_min, m_max = mip.min(), mip.max()
        if m_max - m_min > 0:
            mip_norm = (mip - m_min) / (m_max - m_min + 1e-8)
        else:
            mip_norm = mip
        return mip_norm.astype(np.float32)

    def get_slice(self, axis='axial', index=None) -> np.ndarray:
        if self.volume is None: return None
        shape = self.volume.shape
        
        if axis == 'axial':
            i = index if index is not None else shape[0] // 2
            slc = self.volume[i, :, :]
        elif axis == 'coronal':
            i = index if index is not None else shape[1] // 2
            slc = self.volume[:, i, :]
        elif axis == 'sagittal':
            i = index if index is not None else shape[2] // 2
            slc = self.volume[:, :, i]
        else:
            raise ValueError("Unknown axis")
            
        s_min, s_max = slc.min(), slc.max()
        if s_max - s_min > 0:
            slc_norm = (slc - s_min) / (s_max - s_min + 1e-8)
        else:
            slc_norm = slc
        return slc_norm.astype(np.float32)

    def compute_volume_stats(self) -> dict:
        if self.volume is None: return {}
        shape = self.volume.shape
        voxel_count = shape[0] * shape[1] * shape[2]
        
        return {
            "shape": shape,
            "voxel_count": voxel_count,
            "mean_intensity": round(float(np.mean(self.volume)), 4),
            "std_intensity": round(float(np.std(self.volume)), 4),
            "min_intensity": float(np.min(self.volume)),
            "max_intensity": float(np.max(self.volume)),
            "non_zero_fraction": float(np.mean(self.volume > 0.05))
        }

class IsosurfaceExtractor:
    def __init__(self, iso_value=0.3):
        self.iso_value = iso_value

    def extract_surface(self, volume: np.ndarray) -> dict:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            volume, level=self.iso_value, spacing=(1.0, 1.0, 1.0)
        )
        return {
            "verts": verts,
            "faces": faces,
            "normals": normals,
            "values": values,
            "vertex_count": len(verts),
            "face_count": len(faces)
        }

    def smooth_surface(self, verts: np.ndarray, faces: np.ndarray, iterations=3) -> np.ndarray:
        num_verts = len(verts)
        adj = [set() for _ in range(num_verts)]
        for f in faces:
            adj[f[0]].add(f[1])
            adj[f[0]].add(f[2])
            adj[f[1]].add(f[0])
            adj[f[1]].add(f[2])
            adj[f[2]].add(f[0])
            adj[f[2]].add(f[1])
            
        smoothed_verts = verts.copy()
        
        for _ in range(iterations):
            new_verts = np.zeros_like(smoothed_verts)
            for i in range(num_verts):
                neighbors = list(adj[i])
                if neighbors:
                    new_verts[i] = np.mean(smoothed_verts[neighbors], axis=0)
                else:
                    new_verts[i] = smoothed_verts[i]
            smoothed_verts = new_verts
            
        return smoothed_verts

class VolumeRenderer:
    def render_isosurface(self, surface_data: dict, title="3D Lesion Reconstruction", color_scale="Plasma") -> go.Figure:
        fig = go.Figure(data=[go.Mesh3d(
            x=surface_data["verts"][:, 0],
            y=surface_data["verts"][:, 1],
            z=surface_data["verts"][:, 2],
            i=surface_data["faces"][:, 0],
            j=surface_data["faces"][:, 1],
            k=surface_data["faces"][:, 2],
            intensity=surface_data["verts"][:, 2], # Use Z coordinate to give nice depth coloring
            colorscale=color_scale.lower(),
            opacity=0.85,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=150)
        )])
        
        axis_settings = dict(color='#7F77DD', gridcolor='#7F77DD', zerolinecolor='#7F77DD')
        fig.update_layout(
            paper_bgcolor='#0D0E1A',
            plot_bgcolor='#0D0E1A',
            scene=dict(bgcolor='#0D0E1A', xaxis=axis_settings, yaxis=axis_settings, zaxis=axis_settings),
            title=dict(text=title, font=dict(color='#FFFFFF')),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    def render_volume_slices(self, volume: np.ndarray) -> go.Figure:
        stacker = VolumetricStacker()
        stacker.volume = volume
        
        axial_mip = stacker.get_projection('axial')
        coronal_mip = stacker.get_projection('coronal')
        sagittal_mip = stacker.get_projection('sagittal')
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Axial MIP", "Coronal MIP", "Sagittal MIP"))
        
        fig.add_trace(go.Heatmap(z=axial_mip, colorscale='Greys', showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=coronal_mip, colorscale='Greys', showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=sagittal_mip, colorscale='Greys', showscale=False), row=1, col=3)
        
        fig.update_layout(
            paper_bgcolor='#0D0E1A',
            plot_bgcolor='#0D0E1A',
            font=dict(color='#FFFFFF'),
            margin=dict(l=20, r=20, b=20, t=40)
        )
        
        for i in range(1, 4):
            fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=i)
            fig.update_yaxes(showgrid=False, showticklabels=False, autorange='reversed', row=1, col=i)
            
        return fig
