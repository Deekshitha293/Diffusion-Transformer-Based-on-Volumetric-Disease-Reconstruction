import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

class PatchEmbedder(nn.Module):
    def __init__(self, img_size=256, patch_size=32, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.projection = nn.Linear(patch_size * patch_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, img: np.ndarray) -> torch.Tensor:
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        patches = einops.rearrange(img_tensor, '1 1 (h p1) (w p2) -> 1 (h w) (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        embeddings = self.projection(patches)
        embeddings = embeddings + self.position_embedding
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)

        out = (attn_weights @ v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out, attn_weights.detach().cpu().numpy()

class InterSliceAttention:
    def __init__(self, embed_dim=256, num_heads=8):
        self.embedder = PatchEmbedder(embed_dim=embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.embed_dim = embed_dim

    def compute_slice_embeddings(self, slices: list) -> torch.Tensor:
        embeddings = []
        for img in slices:
            emb = self.embedder.forward(img)
            emb_mean = emb.mean(dim=1, keepdim=True)
            embeddings.append(emb_mean)
        stacked = torch.cat(embeddings, dim=1)
        return stacked

    def compute_inter_slice_attention(self, slices: list) -> dict:
        with torch.no_grad():
            stacked_embs = self.compute_slice_embeddings(slices)
            _, attn_weights = self.attention.forward(stacked_embs)
            mean_attention = attn_weights.mean(axis=1)[0]
            
        return {
            "attention_matrix": mean_attention,
            "num_slices": len(slices),
            "num_heads": self.attention.num_heads
        }

def get_patch_attention_map(slice_img: np.ndarray, embed_dim=256, num_heads=8) -> np.ndarray:
    embedder = PatchEmbedder(embed_dim=embed_dim)
    attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    with torch.no_grad():
        emb = embedder.forward(slice_img)
        _, attn_weights = attention.forward(emb)
    
    mean_query_attn = attn_weights[0].mean(axis=0).mean(axis=0)
    num_patches_per_side = int(np.sqrt(mean_query_attn.shape[0]))
    heat_map = mean_query_attn.reshape(num_patches_per_side, num_patches_per_side)
    heat_map_resized = cv2.resize(heat_map, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    min_val, max_val = heat_map_resized.min(), heat_map_resized.max()
    heat_map_norm = (heat_map_resized - min_val) / (max_val - min_val + 1e-8)
    
    return heat_map_norm.astype(np.float32)
