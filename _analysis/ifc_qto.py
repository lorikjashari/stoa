import re, collections, sys

p = sys.argv[1]
txt = open(p, encoding="latin-1").read()
q = collections.Counter(re.findall(r"IFCQUANTITY(?:LENGTH|AREA|VOLUME|COUNT)\('([^']+)'", txt))
print("quantities:", q.most_common(40))
print("qsets:", collections.Counter(re.findall(r"IFCELEMENTQUANTITY\('[^']*',#\d+,'([^']+)'", txt)).most_common(20))
print("psets:", collections.Counter(re.findall(r"IFCPROPERTYSET\('[^']*',#\d+,'([^']+)'", txt)).most_common(20))
