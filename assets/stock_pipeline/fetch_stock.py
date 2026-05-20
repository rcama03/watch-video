"""
Stock Footage & Image Pipeline
Usage:
    python fetch_stock.py --script "your voiceover script here" --out ./footage
    python fetch_stock.py --script script.txt --out ./footage --images --videos
    python fetch_stock.py --topics "boarding,cabin crew,senior passenger" --out ./footage
"""

import os
import re
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# CONFIG — edit these once, reuse forever
# ---------------------------------------------------------------------------

PEXELS_API_KEY  = os.getenv("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

# Minimum quality thresholds
MIN_VIDEO_WIDTH   = 1920
MIN_VIDEO_HEIGHT  = 1080
MIN_VIDEO_FPS     = 25         # reject slow-motion clips
MIN_IMAGE_WIDTH   = 1920
PREFERRED_FORMAT  = "mp4"

# How many results to fetch per query per source
RESULTS_PER_QUERY = 5


# ---------------------------------------------------------------------------
# AVIATION NICHE KEYWORD MAP
# Maps any script keyword → stock search query on each platform
# Add new rows here whenever you expand to a new sub-topic
# ---------------------------------------------------------------------------

KEYWORD_MAP = {
    # keyword in script      : (pexels query,                          pixabay query)

    # Crew & service
    "flight attendant"       : ("flight attendant cabin service",       "flight+attendant+cabin"),
    "cabin crew"             : ("cabin crew airline service passengers", "cabin+crew+airline"),
    "crew"                   : ("flight attendant aisle passengers",     "flight+crew+service"),
    "stewardess"             : ("stewardess serving passengers",         "stewardess+airplane"),
    "pilot"                  : ("airline pilot cockpit",                 "pilot+cockpit+airplane"),

    # Passengers
    "passenger"              : ("passenger airplane seat window",        "passenger+airplane+seat"),
    "senior"                 : ("elderly senior passenger airplane",     "senior+passenger+travel"),
    "elderly"                : ("elderly person airport travel",         "elderly+airport+travel"),
    "traveler"               : ("traveler airport luggage walking",      "traveler+airport+luggage"),

    # Aircraft interior
    "cabin"                  : ("airplane cabin interior aisle",         "airplane+cabin+interior"),
    "seat"                   : ("airplane seat economy class",           "airplane+seat+economy"),
    "window seat"            : ("passenger window seat airplane",        "airplane+window+seat"),
    "overhead bin"           : ("overhead luggage compartment airplane", "overhead+bin+luggage"),
    "call button"            : ("airplane call button overhead panel",   "airplane+call+button"),
    "oxygen mask"            : ("oxygen mask airplane safety",           "oxygen+mask+airplane"),
    "aisle"                  : ("airplane aisle passengers walking",     "airplane+aisle"),
    "exit"                   : ("emergency exit airplane door",          "airplane+emergency+exit"),
    "galley"                 : ("airplane galley kitchen crew",          "airplane+galley"),
    "tray table"             : ("airplane tray table food",              "tray+table+airplane"),

    # Airport
    "airport"                : ("airport terminal departure gate",       "airport+terminal+gate"),
    "boarding"               : ("passengers boarding aircraft gate",     "airplane+boarding"),
    "gate"                   : ("airport departure gate passengers",     "airport+gate+departure"),
    "jetway"                 : ("airport jetway bridge airplane",        "airport+jetway"),
    "check in"               : ("airport check in counter staff",        "airport+check+in"),
    "security"               : ("airport security check passengers",     "airport+security"),
    "luggage"                : ("airport luggage suitcase travel",       "luggage+suitcase+airport"),
    "carry on"               : ("carry on bag overhead airplane",        "carry+on+bag+travel"),
    "passport"               : ("passport boarding pass airport",        "passport+boarding+pass"),
    "terminal"               : ("airport terminal walking passengers",   "airport+terminal+crowd"),

    # Exterior / aircraft
    "airplane"               : ("airplane aircraft flying sky",          "airplane+aircraft+sky"),
    "aircraft"               : ("commercial aircraft runway airport",    "commercial+aircraft+runway"),
    "runway"                 : ("airplane runway takeoff night",         "airplane+runway+night"),
    "takeoff"                : ("airplane takeoff runway lights",        "airplane+takeoff"),
    "landing"                : ("airplane landing airport runway",       "airplane+landing"),
    "tarmac"                 : ("aircraft tarmac airport ground",        "aircraft+tarmac"),

    # Medical / special assistance
    "wheelchair"             : ("wheelchair airport assistance travel",  "wheelchair+airport"),
    "medical"                : ("medical assistance airport airplane",   "medical+assistance+travel"),
    "oxygen"                 : ("oxygen mask airplane safety demo",      "oxygen+airplane"),

    # Comfort / amenities
    "food"                   : ("airplane meal food service cabin",      "airplane+meal+food"),
    "drink"                  : ("flight attendant drink service",        "flight+attendant+drinks"),
    "snack"                  : ("airplane snack service passengers",     "airplane+snack"),
    "blanket"                : ("airplane blanket pillow comfort",       "airplane+blanket+comfort"),
    "upgrade"                : ("business class airplane luxury seat",   "business+class+airplane"),
    "business class"         : ("business class airplane seat luxury",   "business+class+seat"),
    "first class"            : ("first class airplane suite cabin",      "first+class+airplane"),

    # Emotions / behavior
    "sleep"                  : ("passenger sleeping airplane seat",      "passenger+sleeping+airplane"),
    "reading"                : ("passenger reading book airplane",       "passenger+reading+airplane"),
    "stressed"               : ("stressed traveler airport crowd",       "stressed+traveler+airport"),
    "relaxed"                : ("relaxed passenger airplane window",     "relaxed+passenger+airplane"),
    "smile"                  : ("smiling flight attendant cabin",        "smiling+flight+attendant"),
}

# Fallback query if no keyword matches
FALLBACK_QUERY = ("aviation travel airplane", "aviation+travel+airplane")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _download(url: str, dest: Path, label: str = "") -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {label} -> {dest.name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _extract_keywords(script: str) -> list[str]:
    """Return every KEYWORD_MAP key that appears in the script text."""
    script_lower = script.lower()
    found = []
    for keyword in sorted(KEYWORD_MAP.keys(), key=len, reverse=True):  # longest match first
        if keyword in script_lower and keyword not in found:
            found.append(keyword)
    return found if found else list(KEYWORD_MAP.keys())[:6]  # if empty script, use top 6


# ---------------------------------------------------------------------------
# PEXELS
# ---------------------------------------------------------------------------

def _pexels_videos(query: str, per_page: int) -> list[dict]:
    if not PEXELS_API_KEY:
        return []
    url = (
        "https://api.pexels.com/videos/search"
        f"?query={urllib.parse.quote(query)}"
        f"&orientation=landscape&size=large&per_page={per_page}"
    )
    try:
        data = _http_get(url, {"Authorization": PEXELS_API_KEY})
    except Exception as e:
        print(f"  [Pexels video error] {e}")
        return []

    results = []
    for v in data.get("videos", []):
        # pick best quality file
        files = sorted(v.get("video_files", []),
                       key=lambda f: f.get("width", 0), reverse=True)
        for f in files:
            if (f.get("width", 0) >= MIN_VIDEO_WIDTH
                    and f.get("height", 0) >= MIN_VIDEO_HEIGHT
                    and f.get("file_type", "") == f"video/{PREFERRED_FORMAT}"):
                results.append({
                    "source": "pexels",
                    "type": "video",
                    "url": f["link"],
                    "width": f["width"],
                    "height": f["height"],
                    "duration": v.get("duration", 0),
                    "id": v["id"],
                })
                break
    return results


def _pexels_images(query: str, per_page: int) -> list[dict]:
    if not PEXELS_API_KEY:
        return []
    url = (
        "https://api.pexels.com/v1/search"
        f"?query={urllib.parse.quote(query)}"
        f"&orientation=landscape&size=large&per_page={per_page}"
    )
    try:
        data = _http_get(url, {"Authorization": PEXELS_API_KEY})
    except Exception as e:
        print(f"  [Pexels image error] {e}")
        return []

    results = []
    for p in data.get("photos", []):
        w = p.get("width", 0)
        if w >= MIN_IMAGE_WIDTH:
            results.append({
                "source": "pexels",
                "type": "image",
                "url": p["src"]["original"],
                "width": w,
                "height": p.get("height", 0),
                "id": p["id"],
            })
    return results


# ---------------------------------------------------------------------------
# PIXABAY
# ---------------------------------------------------------------------------

def _pixabay_videos(query: str, per_page: int) -> list[dict]:
    if not PIXABAY_API_KEY:
        return []
    url = (
        "https://pixabay.com/api/videos/"
        f"?key={PIXABAY_API_KEY}"
        f"&q={query}&video_type=film&category=travel"
        f"&min_width={MIN_VIDEO_WIDTH}&per_page={per_page}"
    )
    try:
        data = _http_get(url)
    except Exception as e:
        print(f"  [Pixabay video error] {e}")
        return []

    results = []
    for v in data.get("hits", []):
        large = v.get("videos", {}).get("large", {})
        if large.get("width", 0) >= MIN_VIDEO_WIDTH:
            results.append({
                "source": "pixabay",
                "type": "video",
                "url": large["url"],
                "width": large["width"],
                "height": large["height"],
                "duration": v.get("duration", 0),
                "id": v["id"],
            })
    return results


def _pixabay_images(query: str, per_page: int) -> list[dict]:
    if not PIXABAY_API_KEY:
        return []
    url = (
        "https://pixabay.com/api/"
        f"?key={PIXABAY_API_KEY}"
        f"&q={query}&image_type=photo&orientation=horizontal"
        f"&category=travel&min_width={MIN_IMAGE_WIDTH}&per_page={per_page}"
    )
    try:
        data = _http_get(url)
    except Exception as e:
        print(f"  [Pixabay image error] {e}")
        return []

    results = []
    for p in data.get("hits", []):
        results.append({
            "source": "pixabay",
            "type": "image",
            "url": p["largeImageURL"],
            "width": p.get("imageWidth", 0),
            "height": p.get("imageHeight", 0),
            "id": p["id"],
        })
    return results


# ---------------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------------

def fetch_for_keyword(
    keyword: str,
    out_dir: Path,
    want_videos: bool = True,
    want_images: bool = False,
    per_source: int = RESULTS_PER_QUERY,
) -> list[dict]:
    """Fetch footage/images for one keyword from all sources."""
    pexels_q, pixabay_q = KEYWORD_MAP.get(keyword, FALLBACK_QUERY)
    slug = _slugify(keyword)
    folder = out_dir / slug
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n[{keyword}]")
    manifest = []

    if want_videos:
        for item in _pexels_videos(pexels_q, per_source):
            ext = PREFERRED_FORMAT
            dest = folder / f"pexels_v{item['id']}.{ext}"
            if not dest.exists():
                _download(item["url"], dest, f"pexels video {item['id']}")
                time.sleep(0.3)
            item["local_path"] = str(dest)
            manifest.append(item)

        for item in _pixabay_videos(pixabay_q, per_source):
            dest = folder / f"pixabay_v{item['id']}.mp4"
            if not dest.exists():
                _download(item["url"], dest, f"pixabay video {item['id']}")
                time.sleep(0.3)
            item["local_path"] = str(dest)
            manifest.append(item)

    if want_images:
        for item in _pexels_images(pexels_q, per_source):
            dest = folder / f"pexels_i{item['id']}.jpg"
            if not dest.exists():
                _download(item["url"], dest, f"pexels image {item['id']}")
                time.sleep(0.3)
            item["local_path"] = str(dest)
            manifest.append(item)

        for item in _pixabay_images(pixabay_q, per_source):
            dest = folder / f"pixabay_i{item['id']}.jpg"
            if not dest.exists():
                _download(item["url"], dest, f"pixabay image {item['id']}")
                time.sleep(0.3)
            item["local_path"] = str(dest)
            manifest.append(item)

    return manifest


def run_pipeline(
    script: str = "",
    topics: str = "",
    out_dir: str = "./footage",
    want_videos: bool = True,
    want_images: bool = False,
    per_source: int = RESULTS_PER_QUERY,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve keywords
    if topics:
        keywords = [k.strip() for k in topics.split(",")]
    elif script:
        # load from file if path given
        if os.path.isfile(script):
            script = Path(script).read_text()
        keywords = _extract_keywords(script)
    else:
        print("Provide --script or --topics"); sys.exit(1)

    print(f"Keywords detected: {keywords}")
    print(f"Sources: {'video ' if want_videos else ''}{'images' if want_images else ''}")
    print(f"Output:  {out.resolve()}\n")

    if not PEXELS_API_KEY and not PIXABAY_API_KEY:
        print("ERROR: Set PEXELS_API_KEY and/or PIXABAY_API_KEY environment variables.")
        sys.exit(1)

    full_manifest = {}
    for kw in keywords:
        results = fetch_for_keyword(kw, out, want_videos, want_images, per_source)
        full_manifest[kw] = results

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(full_manifest, indent=2))
    print(f"\nManifest saved: {manifest_path}")

    total = sum(len(v) for v in full_manifest.values())
    print(f"Total assets fetched: {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aviation stock footage pipeline")
    parser.add_argument("--script",  default="", help="Voiceover script text or path to .txt file")
    parser.add_argument("--topics",  default="", help="Comma-separated topics, e.g. 'boarding,cabin crew'")
    parser.add_argument("--out",     default="./footage", help="Output folder (default: ./footage)")
    parser.add_argument("--videos",  action="store_true", default=True,  help="Fetch videos (default: on)")
    parser.add_argument("--images",  action="store_true", default=False, help="Also fetch images")
    parser.add_argument("--n",       type=int, default=RESULTS_PER_QUERY, help="Results per keyword per source")
    args = parser.parse_args()

    run_pipeline(
        script=args.script,
        topics=args.topics,
        out_dir=args.out,
        want_videos=args.videos,
        want_images=args.images,
        per_source=args.n,
    )
