"""
Relevancy Filter for Stock Footage/Images
==========================================
Plug this into your existing pipeline to score and keep only
clips that visually match your voiceover sentence.

Two scoring layers (use one or both):

  Layer 1 — Text match (fast, no GPU needed)
    Embeds voiceover sentence + clip metadata (title/tags/description)
    using sentence-transformers and computes cosine similarity.

  Layer 2 — Visual match (accurate, needs downloaded clip/frame)
    Extracts a middle frame from the clip and scores it against
    the voiceover sentence using OpenAI CLIP.

Install deps once:
    pip install sentence-transformers openai-clip pillow opencv-python-headless

Usage in your existing script:
    from relevancy_filter import score_clips, keep_relevant

    scored = score_clips(
        voiceover="The crew makes a quiet decision about you before you even speak",
        clips=your_api_results_list,   # list of dicts with 'title','tags','description'
        use_visual=False               # set True after downloading clips
    )
    relevant = keep_relevant(scored, min_score=0.40)
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# LAYER 1 — Text-based semantic scoring (fast, free, no GPU)
# ---------------------------------------------------------------------------

def _load_text_model():
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, util
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")


def _clip_text(clip: dict) -> str:
    """Combine all text metadata from a clip dict into one string."""
    parts = [
        clip.get("title", ""),
        clip.get("description", ""),
        " ".join(clip.get("tags", [])) if isinstance(clip.get("tags"), list) else clip.get("tags", ""),
        clip.get("alt", ""),         # Pexels image alt text
        clip.get("url", ""),         # URL often contains descriptive slug
    ]
    return " ".join(p for p in parts if p).lower()


def text_score(voiceover: str, clip: dict, model=None, util=None) -> float:
    """
    Return cosine similarity [0.0 – 1.0] between voiceover sentence
    and clip metadata text.
    """
    if model is None:
        model, util = _load_text_model()

    clip_text = _clip_text(clip)
    if not clip_text.strip():
        return 0.0

    vo_emb   = model.encode(voiceover,  convert_to_tensor=True)
    clip_emb = model.encode(clip_text,  convert_to_tensor=True)
    score    = float(util.cos_sim(vo_emb, clip_emb)[0][0])
    return round(max(0.0, score), 4)


# ---------------------------------------------------------------------------
# LAYER 2 — Visual scoring using CLIP (accurate, needs local file)
# ---------------------------------------------------------------------------

def _load_clip_model():
    try:
        import clip, torch
        from PIL import Image
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except ImportError:
        raise ImportError("Run: pip install openai-clip pillow")


def _extract_middle_frame(video_path: str) -> Optional[np.ndarray]:
    """Extract the middle frame of a video as a numpy array."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def visual_score(voiceover: str, local_path: str, model=None, preprocess=None, device=None) -> float:
    """
    Return CLIP similarity [0.0 – 1.0] between voiceover sentence
    and the middle frame of the clip at local_path.
    Works for both video files and image files.
    """
    import torch
    from PIL import Image

    if model is None:
        model, preprocess, device = _load_clip_model()

    path = Path(local_path)
    if not path.exists():
        return 0.0

    # Load frame
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        frame = _extract_middle_frame(str(path))
        if frame is None:
            return 0.0
        pil_image = Image.fromarray(frame)
    else:
        pil_image = Image.open(path).convert("RGB")

    img_input  = preprocess(pil_image).unsqueeze(0).to(device)
    text_input = torch.cat([__import__('clip').tokenize(voiceover[:77])]).to(device)

    with torch.no_grad():
        img_feat  = model.encode_image(img_input)
        txt_feat  = model.encode_text(text_input)
        img_feat  /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat  /= txt_feat.norm(dim=-1, keepdim=True)
        score = float((img_feat @ txt_feat.T)[0][0])

    return round(max(0.0, score), 4)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINTS — plug these into your existing script
# ---------------------------------------------------------------------------

def score_clips(
    voiceover: str,
    clips: list[dict],
    use_visual: bool = False,
    text_weight: float = 0.5,
    visual_weight: float = 0.5,
) -> list[dict]:
    """
    Score every clip against the voiceover sentence.
    Adds 'text_score', 'visual_score', 'final_score' keys to each clip dict.
    Returns list sorted by final_score descending.

    Args:
        voiceover:     One sentence from your voiceover script.
        clips:         List of clip dicts from Pexels/Pixabay API or your existing results.
        use_visual:    True = also run CLIP on local_path (clip must be downloaded first).
        text_weight:   Weight for text similarity (0.0–1.0).
        visual_weight: Weight for visual similarity (0.0–1.0). Only used if use_visual=True.
    """
    model, util = _load_text_model()
    clip_model = preprocess = device = None

    if use_visual:
        clip_model, preprocess, device = _load_clip_model()

    scored = []
    for clip in clips:
        t_score = text_score(voiceover, clip, model, util)

        if use_visual and clip.get("local_path"):
            v_score = visual_score(voiceover, clip["local_path"], clip_model, preprocess, device)
            final   = round(text_weight * t_score + visual_weight * v_score, 4)
        else:
            v_score = None
            final   = t_score

        scored.append({
            **clip,
            "text_score":   t_score,
            "visual_score": v_score,
            "final_score":  final,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


def keep_relevant(scored_clips: list[dict], min_score: float = 0.35, top_n: int = None) -> list[dict]:
    """
    Filter to only clips above min_score.
    Optionally cap at top_n results.

    Recommended thresholds:
        0.40+  → very tight match (use for talking-head b-roll sync)
        0.30+  → good match     (use for general scene coverage)
        0.20+  → loose match    (fallback if pool is small)
    """
    filtered = [c for c in scored_clips if c["final_score"] >= min_score]
    if top_n:
        filtered = filtered[:top_n]
    return filtered


def score_and_filter(
    voiceover: str,
    clips: list[dict],
    min_score: float = 0.35,
    top_n: int = 3,
    use_visual: bool = False,
) -> list[dict]:
    """One-liner convenience wrapper: score + filter in a single call."""
    scored = score_clips(voiceover, clips, use_visual=use_visual)
    return keep_relevant(scored, min_score=min_score, top_n=top_n)


# ---------------------------------------------------------------------------
# VOICEOVER SEGMENTER
# Splits full script into sentences so each sentence gets its own clips
# ---------------------------------------------------------------------------

def split_voiceover(script: str) -> list[str]:
    """
    Split a full voiceover script into individual sentences.
    Each sentence will be matched against its own set of clips.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', script.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# DEMO / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate what your existing script returns from API
    mock_clips = [
        {"title": "Flight attendant serving food to passengers in airplane cabin",
         "tags": ["airline", "cabin crew", "food service", "airplane"],
         "description": "A flight attendant walks through the aisle serving meals"},

        {"title": "Businessman working on laptop at airport terminal",
         "tags": ["airport", "business", "laptop", "terminal"],
         "description": "Man working on computer before flight"},

        {"title": "Senior elderly woman looking out airplane window",
         "tags": ["senior", "passenger", "window", "airplane", "elderly"],
         "description": "Old woman sitting in window seat watching sky"},

        {"title": "Flight crew checking passenger manifest before boarding",
         "tags": ["crew", "boarding", "flight", "check", "manifest"],
         "description": "Crew reviewing passenger list at gate"},

        {"title": "Empty overhead bin in airplane cabin",
         "tags": ["overhead", "bin", "luggage", "cabin", "airplane"],
         "description": "Open overhead compartment with no bags"},
    ]

    voiceover = "The crew makes a quiet decision about you before you even speak to them"

    print(f"Voiceover: \"{voiceover}\"\n")
    scored = score_clips(voiceover, mock_clips, use_visual=False)

    print(f"{'Score':>6}  Title")
    print("-" * 70)
    for c in scored:
        print(f"{c['final_score']:>6.3f}  {c['title'][:65]}")

    print("\nKeeping only score >= 0.35:")
    relevant = keep_relevant(scored, min_score=0.35)
    for c in relevant:
        print(f"  [{c['final_score']:.3f}] {c['title'][:65]}")
