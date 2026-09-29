import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import PurePosixPath

FOLDERS = {
    "KoIn10": "https://drive.google.com/drive/folders/178c7uJM99eY87OZJQAy_Vn6u5xgAezwc?usp=sharing",
    "KoIn50": "https://drive.google.com/drive/folders/14V2QCmqjrMXgasbnuZ0NpnLzWU2621fC?usp=sharing",
    "KoIn100": "https://drive.google.com/drive/folders/10t3fAwlNV764pzHz1crinezgDfEsU1i5?usp=sharing",
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_json_probe(url: str):
    # gdown 6.x: --folder tells it to enumerate a Drive folder; --json returns
    # the planned file URL/path list without downloading the images.
    cmds = [
        [sys.executable, "-m", "gdown", "--folder", "--json", url],
        ["gdown", "--folder", "--json", url],
    ]
    errors = []
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
            if p.returncode != 0:
                errors.append({"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-3000:], "stderr": p.stderr[-5000:]})
                continue
            text = p.stdout.strip()
            starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
            if starts:
                text = text[min(starts):]
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("files", "items", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise TypeError(f"unexpected JSON type: {type(data).__name__}")
            return data, {"command": cmd, "stderr_tail": p.stderr[-2000:]}
        except Exception as e:
            errors.append({"cmd": cmd, "error": repr(e)})
    raise RuntimeError(json.dumps(errors, ensure_ascii=False))


def normalize_entry(entry):
    if isinstance(entry, str):
        return {"path": entry, "url": None}
    if not isinstance(entry, dict):
        return {"path": str(entry), "url": None}
    return {
        "path": entry.get("path") or entry.get("name") or entry.get("filename") or "",
        "url": entry.get("url") or entry.get("link") or entry.get("id"),
        "raw": entry,
    }


def summarize(entries):
    norm = [normalize_entry(e) for e in entries]
    paths = [n["path"] for n in norm if n["path"]]
    image_paths = [p for p in paths if PurePosixPath(p).suffix.lower() in IMAGE_EXT]
    ext_counts = Counter(PurePosixPath(p).suffix.lower() or "<none>" for p in paths)
    depth_counts = Counter(len(PurePosixPath(p).parts) for p in paths)

    class_counts = Counter()
    for p in image_paths:
        parts = PurePosixPath(p).parts[:-1]
        numeric = [x for x in parts if x.isdigit() and len(x) <= 4]
        if numeric:
            class_counts[numeric[-1].zfill(4)] += 1

    dirs_by_depth = defaultdict(Counter)
    for p in paths:
        parts = PurePosixPath(p).parts
        for i, part in enumerate(parts[:-1]):
            dirs_by_depth[str(i)][part] += 1

    return {
        "entry_count": len(entries),
        "path_count": len(paths),
        "image_count": len(image_paths),
        "extension_counts": dict(ext_counts.most_common()),
        "path_depth_counts": dict(sorted(depth_counts.items())),
        "detected_numeric_classes": len(class_counts),
        "numeric_class_image_counts": dict(sorted(class_counts.items())),
        "top_directories_by_depth": {
            d: c.most_common(30) for d, c in sorted(dirs_by_depth.items(), key=lambda kv: int(kv[0]))
        },
        "sample_paths": paths[:120],
        "sample_entries": norm[:30],
    }


def main():
    report = {
        "source": "KoIn official Google Drive folders linked from dukong1/KoIn_Benchmark_Dataset",
        "downloaded_images": 0,
        "mode": "metadata-only gdown JSON probe",
        "datasets": {},
    }
    for name, url in FOLDERS.items():
        print(f"=== probing {name} ===", flush=True)
        try:
            entries, meta = run_json_probe(url)
            summary = summarize(entries)
            summary["folder_url"] = url
            summary["probe_meta"] = meta
            report["datasets"][name] = summary
            print(json.dumps({
                "name": name,
                "entries": summary["entry_count"],
                "images": summary["image_count"],
                "classes": summary["detected_numeric_classes"],
                "sample_paths": summary["sample_paths"][:12],
            }, ensure_ascii=False, indent=2), flush=True)
        except Exception as e:
            report["datasets"][name] = {"folder_url": url, "error": repr(e)}
            print(f"probe failed for {name}: {e!r}", flush=True)

    with open("koin_probe.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ok = [v for v in report["datasets"].values() if "error" not in v]
    if not ok:
        raise SystemExit("All KoIn Drive probes failed")


if __name__ == "__main__":
    main()
