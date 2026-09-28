import hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image
import train_real_faces as m

m.PEOPLE = [
    ("taylor_swift", "Taylor Swift", ["exact-list"]),
    ("zendaya", "Zendaya", ["exact-list"]),
    ("tom_holland", "Tom Holland", ["exact-list"]),
]

FILES = {
    "taylor_swift": [
        "File:Swift, Taylor (2007).jpg",
        "File:Swift, Taylor (2007) cropped.jpg",
        "File:Swift, Taylor (2007) cropped 2.JPG",
        "File:Taylor Swift at Yahoo 2007.jpg",
        "File:191125 Taylor Swift at the 2019 American Music Awards.png",
        "File:191125 Taylor Swift at the 2019 American Music Awards (2).png",
        "File:191125 Taylor Swift at the 2019 American Music Awards (cropped).png",
        "File:Taylor Swift 2019 by Glenn Francis.jpg",
    ],
    "zendaya": [
        "File:Zendaya - 2019 by Glenn Francis.jpg",
        "File:Zendaya 2019 by Glenn Francis (cropped).jpg",
        "File:Zendaya 2019 by Glenn Francis.jpg",
        "File:Zendaya Full Length - 2019 by Glenn Francis (cropped).jpg",
        "File:Zendaya Full Length - 2019 by Glenn Francis.jpg",
        "File:Zendaya 2024 (cropped).jpg",
        "File:Zendaya 2024.jpg",
        "File:Zendaya in 2024.jpg",
        "File:Zendaya 2026 (cropped).jpg",
        "File:Zendaya 2026.jpg",
    ],
    "tom_holland": [
        "File:Tom Holland (28035716544).jpg",
        "File:Tom Holland (28036487013).jpg",
        "File:Tom Holland (28620384206).jpg",
        "File:Tom Holland (28652884235) (cropped).jpg",
        "File:Tom Holland (28652884235).jpg",
        "File:Tom Holland (28652888235).jpg",
        "File:Tom Holland (28652891355) (cropped).jpg",
        "File:Tom Holland (28652891355).jpg",
        "File:Tom Holland (28652895005).jpg",
        "File:Tom Holland by Gage Skidmore.jpg",
        "File:Tom Holland MTV 2018 (01).jpg",
        "File:Tom Holland MTV 2018 (02) (cropped).jpg",
        "File:Tom Holland MTV 2018 (02).jpg",
    ],
}

NAMES = {i:n for i,n,_ in m.PEOPLE}

def collect_exact():
    manifest=[]
    for ident,name,_ in m.PEOPLE:
        d=m.RAW/ident
        d.mkdir(parents=True, exist_ok=True)
        n=0
        seen=set()
        for title in FILES[ident]:
            try:
                ii=m.image_info(title)
                if not ii:
                    print("missing:", title); continue
                mime=(ii.get("mime") or "").lower()
                if mime not in {"image/jpeg","image/png","image/webp"}:
                    continue
                meta=ii.get("extmetadata") or {}
                ok,_=m.license_ok(meta)
                if not ok:
                    print("license rejected:", title); continue
                url=ii.get("thumburl") or ii.get("url")
                if not url: continue
                content=m.download(url)
                tmp=d/"_tmp_image"
                tmp.write_bytes(content)
                with Image.open(tmp) as im:
                    rgb=np.asarray(im.convert("RGB"))
                tmp.unlink(missing_ok=True)
                crop=m.largest_face_crop(rgb)
                if crop is None:
                    print("no face:", title); continue
                h=hashlib.sha256(crop.tobytes()).hexdigest()
                if h in seen: continue
                seen.add(h)
                Image.fromarray(crop).save(d/f"{n:02d}.jpg", quality=94)
                manifest.append({
                    "identity":ident,"display_name":name,"file_title":title,
                    "source_url":ii.get("descriptionurl"),
                    "license":(meta.get("LicenseShortName") or {}).get("value",""),
                    "license_url":(meta.get("LicenseUrl") or {}).get("value",""),
                    "artist":(meta.get("Artist") or {}).get("value",""),
                    "face_crop":"largest detected frontal face",
                    "selection":"explicit verified Commons file list",
                })
                n+=1
                print(f"accepted {name}: {n} :: {title}")
                time.sleep(0.12)
            except Exception as e:
                print("skip", title, repr(e))
        if n < 4:
            raise RuntimeError(f"Insufficient verified face crops for {name}: {n}")
    (m.OUT/"sources.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest

if __name__ == "__main__":
    collect_exact()
    m.train()
