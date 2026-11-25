import base64
import hashlib
import zlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from chromboundary.data import atomic_path

NS = "{http://psi.hupo.org/ms/mzml}"


def fetch_file(url, path, expected):
    path = Path(path)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return path
    with urlopen(url, timeout=90) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"File checksum mismatch for {path.name}")
    with atomic_path(path) as temporary:
        temporary.write_bytes(data)
    return path


def arrays(node, time_dtype=np.float64):
    data = {}
    for array in node.findall(f"{NS}binaryDataArrayList/{NS}binaryDataArray"):
        params = {cv.get("accession"): cv for cv in array.findall(NS + "cvParam")}
        if "MS:1000595" in params:
            kind = "time"
        elif "MS:1000515" in params:
            kind = "intensity"
        else:
            continue
        (dtype,) = [d for code, d in (("MS:1000521", "<f4"), ("MS:1000523", "<f8")) if code in params]
        raw = base64.b64decode("".join(array.findtext(NS + "binary", "").split()), validate=True)
        if "MS:1000574" in params:
            raw = zlib.decompress(raw)
        elif "MS:1000576" not in params:
            raise ValueError("Unsupported mzML compression")
        values = np.frombuffer(raw, dtype=dtype)
        if kind == "time":
            unit = params["MS:1000595"].get("unitAccession")
            if time_dtype is not None:
                values = values.astype(time_dtype, copy=False)
            values = values * {"UO:0000031": 60, "UO:0000010": 1}[unit]
        if len(values) != int(node.get("defaultArrayLength")) or kind in data:
            raise ValueError(f"Invalid numerical array in {node.get('id')}")
        data[kind] = values
    return data["time"], data["intensity"]
