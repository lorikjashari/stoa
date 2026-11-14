import collections, sys

p = sys.argv[1]
L = [x.strip() for x in open(p, encoding="utf-8", errors="replace").read().split("\n")]
print("lines", len(L))
if "$ACADVER" in L:
    print("version", L[L.index("$ACADVER") + 2])
if "$INSUNITS" in L:
    print("insunits", L[L.index("$INSUNITS") + 2])
s = next(i for i in range(len(L) - 2) if L[i] == "SECTION" and L[i + 2] == "ENTITIES") + 3
ents = collections.Counter(); lay = collections.Counter(); el = collections.Counter()
cur = None
curlayer = None
texts = []
for i in range(s, len(L) - 1, 2):
    code, val = L[i], L[i + 1]
    if code == "0":
        cur = val; ents[val] += 1
    elif code == "8":
        lay[val] += 1; el[(cur, val)] += 1; curlayer = val
    elif code == "1":
        texts.append((curlayer, val))
print(ents.most_common())
print(lay.most_common())
for k, v in sorted(el.items()):
    print(k, v)
print(texts[:150])
