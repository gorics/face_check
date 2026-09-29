import json
import re
from collections import deque

import requests
from gdown.download_folder import _GoogleDriveFile, _parse_embedded_folder_view

FOLDERS = {
    "KoIn10": "https://drive.google.com/drive/folders/178c7uJM99eY87OZJQAy_Vn6u5xgAezwc?usp=sharing",
    "KoIn50": "https://drive.google.com/drive/folders/14V2QCmqjrMXgasbnuZ0NpnLzWU2621fC?usp=sharing",
    "KoIn100": "https://drive.google.com/drive/folders/10t3fAwlNV764pzHz1crinezgDfEsU1i5?usp=sharing",
}


def folder_id(url):
    m = re.search(r"/folders/([^/?#]+)", url)
    if not m:
        raise ValueError(url)
    return m.group(1)


def is_class_name(name):
    # KoIn README documents identity directories as 0000 ... 0099.
    return bool(re.fullmatch(r"\d{1,4}", name.strip()))


def list_immediate(sess, fid):
    result = _parse_embedded_folder_view(
        sess=sess,
        folder_id=fid,
        verify=True,
        timeout=30,
    )
    if result is None:
        raise RuntimeError(f"could not parse folder {fid}")
    name, children = result
    return name, children


def scan_dataset(label, url, max_depth=8):
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
    })
    root_id = folder_id(url)
    q = deque([(root_id, "", 0)])
    visited = set()
    folder_nodes = []
    class_folders = []
    errors = []

    while q:
        fid, parent_path, depth = q.popleft()
        if fid in visited or depth > max_depth:
            continue
        visited.add(fid)
        try:
            folder_name, children = list_immediate(sess, fid)
        except Exception as e:
            errors.append({"folder_id": fid, "path": parent_path, "error": repr(e)})
            continue

        here = f"{parent_path}/{folder_name}".strip("/")
        folder_children = []
        file_children = 0
        for child in children:
            cid, cname, ctype = child[:3]
            if ctype == _GoogleDriveFile.TYPE_FOLDER:
                folder_children.append({"id": cid, "name": cname})
            else:
                file_children += 1

        folder_nodes.append({
            "id": fid,
            "path": here,
            "depth": depth,
            "subfolders": len(folder_children),
            "immediate_files": file_children,
        })
        print(f"[{label}] depth={depth} {here!r}: folders={len(folder_children)} files={file_children}", flush=True)

        for child in folder_children:
            cpath = f"{here}/{child['name']}".strip("/")
            if is_class_name(child["name"]):
                class_folders.append({
                    "class": child["name"].zfill(4),
                    "id": child["id"],
                    "path": cpath,
                })
                print(f"[{label}] CLASS {child['name'].zfill(4)} -> {child['id']}", flush=True)
            else:
                q.append((child["id"], here, depth + 1))

    # Dedupe by class/path; a dataset may expose normal/test branches separately.
    unique = []
    seen = set()
    for row in class_folders:
        key = (row["class"], row["id"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    return {
        "folder_url": url,
        "root_id": root_id,
        "visited_folders": len(visited),
        "folder_nodes": folder_nodes,
        "class_folder_count": len(unique),
        "class_folders": sorted(unique, key=lambda x: (x["path"], x["class"])),
        "errors": errors,
    }


def main():
    report = {
        "source": "Official KoIn Google Drive links from dukong1/KoIn_Benchmark_Dataset",
        "mode": "folder-only embeddedfolderview probe; no celebrity images downloaded",
        "downloaded_images": 0,
        "datasets": {},
    }
    for label, url in FOLDERS.items():
        print(f"=== {label} folder-only probe ===", flush=True)
        try:
            report["datasets"][label] = scan_dataset(label, url)
        except Exception as e:
            report["datasets"][label] = {"folder_url": url, "error": repr(e)}

    with open("koin_probe.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===", flush=True)
    for label, d in report["datasets"].items():
        print(label, {
            "visited_folders": d.get("visited_folders"),
            "class_folder_count": d.get("class_folder_count"),
            "error": d.get("error"),
        }, flush=True)

    if not any(d.get("class_folder_count", 0) for d in report["datasets"].values()):
        raise SystemExit("No KoIn identity folders found")


if __name__ == "__main__":
    main()
