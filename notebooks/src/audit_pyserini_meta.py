# Pyserini 1.6.0 registry entry for wikipedia-dpr-100w, read from source with ast (no Java, no index download)
import ast, json, urllib.request
SRC = "https://raw.githubusercontent.com/castorini/pyserini/pyserini-1.6.0/pyserini/prebuilt_index_info.py"
src = urllib.request.urlopen(SRC, timeout=60).read().decode()
entry = None
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "wikipedia-dpr-100w" and isinstance(v, ast.Dict):
                entry = ast.literal_eval(v)
                break
    if entry:
        break
print(json.dumps(entry, indent=2))
url = entry["urls"][0]
req = urllib.request.Request(url, method="HEAD")   # headers only; nothing downloaded
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("HEAD", r.status, "Content-Length:", r.headers.get("Content-Length"))
except Exception as e:
    print("HEAD failed:", type(e).__name__, e)
