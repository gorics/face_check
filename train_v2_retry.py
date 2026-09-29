import time
import train_v2 as m

# Add verified portrait-heavy fallbacks so each identity still has enough
# usable samples after face detection, license filtering, and deduplication.
m.FILES["emma_watson"].extend([
    "File:Emma Watson (5930414886).jpg",
    "File:Emma Watson 2011.jpg",
    "File:Emma Watson at Avery Fisher Hall-Lincoln Center.jpg",
    "File:Emma Watson, 2011.jpg",
    "File:Emma Watson 2012 Shankbone 2.JPG",
    "File:Emma Watson 2012 Shankbone.JPG",
    "File:Emma Watson 2012.jpg",
    "File:Emma Watson, 2012.jpg",
])

m.FILES["margot_robbie"].extend([
    "File:Margot Robbie at Somerset House in 2013.jpg",
    "File:Margot Robbie (28129125529).jpg",
    "File:Margot Robbie (28129125629).jpg",
    "File:Margot Robbie 2018.png",
    "File:29th Critics Choice Awards - Margot Robbie 1.jpg",
    "File:29th Critics Choice Awards - Margot Robbie 3 (cropped).jpg",
])

m.FILES["chris_hemsworth"].extend([
    "File:Chris Hemsworth (7400859836).jpg",
    "File:Chris Hemsworth (7400860530).jpg",
    "File:Chris Hemsworth 2, 2012.jpg",
    "File:Chris Hemsworth 2012.jpg",
    "File:Chris Hemsworth, April 2012.jpg",
    "File:Chris Hemsworth at the 2024 Cannes Film Festival.jpg",
])

m.FILES["ryan_gosling"].extend([
    "File:Ryan Gosling (36201256705) (cropped).jpg",
    "File:Ryan Gosling by Gage Skidmore.jpg",
    "File:Ryan Gosling at SSIFF 2018 (3).jpg",
    "File:Ryan Gosling at SSIFF 2018 (5).jpg",
    "File:Ryan Gosling in 2018.jpg",
])


def faster_retry_get(url, *, params=None, attempts=4, timeout=35):
    last = None
    for k in range(attempts):
        try:
            r = m.session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = 1.5 + 1.25 * k
                print(f"429 short retry in {wait:.2f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            # A small pacing delay is cheaper than long 429 backoffs.
            time.sleep(0.16)
            return r
        except Exception as e:
            last = e
            time.sleep(0.5 + 0.5 * k)
    if last is not None:
        raise last
    raise RuntimeError("request failed after rate-limit retries")


m.retry_get = faster_retry_get

if __name__ == "__main__":
    counts = m.collect()
    m.train(counts)
