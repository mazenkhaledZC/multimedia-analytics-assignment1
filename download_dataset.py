"""
Download NASA Technical Reports from NTRS (ntrs.nasa.gov).
These reports are rich in diagrams, charts, tables, and figures —
ideal for multi-modal RAG with ColPali.
"""

import os
import time
import requests

SAVE_DIR = "data/pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Curated NASA technical reports spanning diverse topics:
# aeronautics, robotics, materials science, propulsion, earth science
NASA_REPORT_IDS = [
    "20240002097",  # Autonomous systems
    "20230016875",  # Aircraft structures
    "20220006691",  # Propulsion performance
    "20210025626",  # Space robotics
    "20200001067",  # Climate/earth science
    "20190001345",  # Aerodynamics
    "20180006200",  # Advanced materials
    "20170001463",  # Flight dynamics
    "20160003254",  # Mission planning
    "20150006789",  # Thermal systems
]

NTRS_PDF_URL = "https://ntrs.nasa.gov/api/citations/{id}/downloads/{id}.pdf"
NTRS_META_URL = "https://ntrs.nasa.gov/api/citations/{id}"


def get_report_metadata(report_id: str) -> dict:
    resp = requests.get(NTRS_META_URL.format(id=report_id), timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return {}


def download_report(report_id: str) -> str | None:
    meta = get_report_metadata(report_id)

    # Try to get the actual download link from metadata
    downloads = []
    if meta:
        downloads = meta.get("_downloads", []) or meta.get("downloads", [])

    save_path = os.path.join(SAVE_DIR, f"nasa_{report_id}.pdf")
    if os.path.exists(save_path):
        print(f"  [skip] {report_id} already downloaded")
        return save_path

    # Try NTRS direct download
    url = NTRS_PDF_URL.format(id=report_id)
    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 200 and "pdf" in resp.headers.get("content-type", "").lower():
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size_kb = os.path.getsize(save_path) / 1024
            print(f"  [ok] {report_id} → {size_kb:.0f} KB")
            return save_path
    except Exception as e:
        print(f"  [fail] {report_id}: {e}")
    return None


def search_and_download(query: str = "aeronautics technical report", max_docs: int = 15):
    """Search NTRS for reports and download PDFs."""
    search_url = "https://ntrs.nasa.gov/api/citations/search"
    params = {
        "q": query,
        "rows": max_docs,
        "start": 0,
        "sort": "relevance",
    }
    try:
        resp = requests.get(search_url, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        print(f"Found {len(results)} results for '{query}'")
        downloaded = []
        for doc in results:
            doc_id = doc.get("id") or doc.get("_id")
            if not doc_id:
                continue
            title = doc.get("title", "")[:60]
            print(f"\n  Downloading: [{doc_id}] {title}")
            path = download_report(str(doc_id))
            if path:
                downloaded.append(path)
            time.sleep(0.5)  # be polite to the server
        return downloaded
    except Exception as e:
        print(f"Search failed: {e}")
        return []


if __name__ == "__main__":
    print("=== Downloading NASA Technical Reports ===\n")

    # Search across multiple technical domains for diversity
    queries = [
        "propulsion system performance analysis charts",
        "aerodynamics computational fluid dynamics",
        "spacecraft thermal control system",
        "autonomous robotics Mars rover",
        "aircraft structural analysis fatigue",
    ]

    all_downloaded = []
    for q in queries:
        print(f"\n[Query] {q}")
        paths = search_and_download(query=q, max_docs=4)
        all_downloaded.extend(paths)
        time.sleep(1)

    # Deduplicate
    all_downloaded = list(set(all_downloaded))
    print(f"\n=== Downloaded {len(all_downloaded)} unique PDFs ===")
    for p in sorted(all_downloaded):
        size_kb = os.path.getsize(p) / 1024
        print(f"  {p} ({size_kb:.0f} KB)")
