import numpy as np
import cv2
import nibabel as nib
from scipy.ndimage import gaussian_filter, shift
from scipy.signal import correlate2d
import faiss
import tempfile
import os

class SlicePreprocessor:
    def __init__(self, target_size=(256, 256), noise_sigma=1.0):
        self.target_size = target_size
        self.noise_sigma = noise_sigma
        self.processed_slices = []
        self.metadata = {}

    def load_image(self, file_bytes, filename) -> np.ndarray:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        elif filename.lower().endswith(('.nii', '.nii.gz')):
            suffix = '.nii.gz' if filename.lower().endswith('.nii.gz') else '.nii'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_name = tmp.name
            
            try:
                img_nii = nib.load(tmp_name)
                data = img_nii.get_fdata()
                # extract middle axial slice (assuming 3rd dim is axial)
                if len(data.shape) >= 3:
                    mid_idx = data.shape[2] // 2
                    img = data[:, :, mid_idx]
                else:
                    img = data
            finally:
                os.remove(tmp_name)
            return img
        return None

    def normalize_slice(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32)
        min_val, max_val = np.min(img_float), np.max(img_float)
        img_norm = (img_float - min_val) / (max_val - min_val + 1e-8)
        return img_norm.astype(np.float32)

    def denoise_slice(self, img: np.ndarray) -> np.ndarray:
        blurred = gaussian_filter(img, sigma=self.noise_sigma)
        return blurred.astype(np.float32)

    def align_slices(self, slices: list) -> list:
        if not slices:
            return []
        aligned = [slices[0]]
        cumulative_shift_y = 0
        cumulative_shift_x = 0
        
        for i in range(1, len(slices)):
            prev = slices[i-1]
            curr = slices[i]
            
            # approximate normalized cross-correlation
            c_curr = curr - np.mean(curr)
            c_prev = prev - np.mean(prev)
            
            corr = correlate2d(c_curr, c_prev, mode='same', boundary='symm')
            
            y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
            cy, cx = curr.shape[0] // 2, curr.shape[1] // 2
            
            shift_y_rel = cy - y_peak
            shift_x_rel = cx - x_peak
            
            cumulative_shift_y += shift_y_rel
            cumulative_shift_x += shift_x_rel
            
            aligned_slice = shift(curr, shift=(cumulative_shift_y, cumulative_shift_x), mode='nearest')
            aligned.append(aligned_slice)
            
        return aligned

    def extract_embedding(self, img: np.ndarray) -> np.ndarray:
        flat = img.flatten()
        np.random.seed(42)
        proj_matrix = np.random.randn(flat.shape[0], 128).astype(np.float32)
        reduced = np.dot(flat, proj_matrix)
        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm
        return reduced.astype(np.float32)

    def process_batch(self, file_list: list, do_align: bool = True) -> dict:
        processed_slices = []
        filenames = []
        for filename, file_bytes in file_list:
            img = self.load_image(file_bytes, filename)
            if img is not None:
                img_norm = self.normalize_slice(img)
                img_denoised = self.denoise_slice(img_norm)
                processed_slices.append(img_denoised)
                filenames.append(filename)
                
        if do_align:
            aligned_slices = self.align_slices(processed_slices)
        else:
            aligned_slices = processed_slices
            
        embeddings = np.array([self.extract_embedding(s) for s in aligned_slices])
        
        return {
            "slices": aligned_slices,
            "embeddings": embeddings,
            "filenames": filenames,
            "count": len(aligned_slices),
            "shape": self.target_size
        }

class FAISSSliceIndex:
    def __init__(self, embedding_dim=128):
        self.dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.slice_map = []

    def build_index(self, embeddings: np.ndarray, filenames: list):
        if len(embeddings) > 0:
            embs = np.asarray(embeddings, dtype=np.float32)
            if len(embs.shape) == 1:
                embs = embs.reshape(1, -1)
            self.index.add(embs)
            self.slice_map.extend(filenames)
            print(f"FAISS index built with {self.index.ntotal} slices.")

    def query(self, query_embedding: np.ndarray, top_k=5) -> list:
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        D, I = self.index.search(q, k)
        results = []
        for i in range(k):
            idx = I[0][i]
            if idx != -1 and idx < len(self.slice_map):
                results.append({
                    "rank": i + 1,
                    "filename": self.slice_map[idx],
                    "distance": float(D[0][i])
                })
        return results

    def get_index_stats(self) -> dict:
        return {
            "total_slices": self.index.ntotal,
            "embedding_dim": self.dim,
            "index_type": "FlatL2"
        }

def generate_sample_slices(n=20, size=(256, 256)) -> list:
    np.random.seed(0)
    slices = []
    for _ in range(n):
        img = np.zeros(size, dtype=np.float32)
        cx, cy = size[1] // 2, size[0] // 2
        rx = np.random.randint(30, 81)
        ry = np.random.randint(30, 81)
        
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        noise = np.random.normal(0, 0.05, size).astype(np.float32)
        img += noise
        
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)
        slices.append(img.astype(np.float32))
    return slices
