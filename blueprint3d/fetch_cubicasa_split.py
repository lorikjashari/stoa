"""Fetch CubiCasa5K's train/val/test split lists from the 5 GB Zenodo zip without downloading the whole zip.

Uses HTTP range requests: reads the zip's central directory, then only the small *.txt split files.

    py -I fetch_cubicasa_split.py [--out third_party/cubicasa_split] [--member path/inside/zip ...]
"""
import argparse, io, os, sys, urllib.request, zipfile

URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url  # follow redirects once
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "third_party", "cubicasa_split"))
    ap.add_argument("--member", nargs="*", default=[], help="extra files to extract (e.g. one image to verify naming)")
    a = ap.parse_args()
    f = HttpRangeFile(URL)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = z.namelist()
    want = [n for n in names if n.endswith((".txt",)) and n.count("/") <= 2] + a.member
    os.makedirs(a.out, exist_ok=True)
    for n in want:
        dest = os.path.join(a.out, n.replace("/", "__"))
        with z.open(n) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        print(f"extracted {n} -> {dest} ({os.path.getsize(dest)} bytes)")
    print(f"zip size {f.size / 1e9:.2f} GB, entries {len(names)}, downloaded {f.fetched / 1e6:.1f} MB")
    print("sample entries:", [n for n in names if n.endswith(".png")][:4])


if __name__ == "__main__":
    sys.exit(main())
