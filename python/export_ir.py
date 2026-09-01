"""Exports the trained model to the v1 IR: graph.json + weights.bin (w/ graph.txt).

Merged weight and IR format export script for .bin and .json file output, retaining same
offset, loop, and manifest for syncronized output.

See docs/IR_SPEC.md for the format this produces.
"""

import hashlib
import json
import os

import numpy as np
import torch
from torch import nn

import mlp_model

ALIGN = 64
IR_VERSION = 1
WEIGHTS_FILE = "weights.bin"


def align_up(x, a):
    return (x + a - 1) // a * a


def build_weights(model):
    """Serialize every Linear weight/bias into one aligned blob, building the manifest.

    Returns (blob, manifest). byte_offset is the section start after padding; byte_len is
    the *unpadded* payload. Padding lives strictly between sections and is never covered
    by any byte_len -- that is the invariant the C++ loader validates.
    """
    blob = bytearray()
    manifest = []
    offset = 0

    for name, layer in model.graph_layers():
        if not isinstance(layer, nn.Linear):
            continue

        for suffix, param in (("weight", layer.weight), ("bias", layer.bias)):
            # ascontiguousarray guards against a non-contiguous view sneaking in later;
            # "<f4" makes the little-endian float32 contract explicit rather than an
            # accident of the host architecture.
            arr = np.ascontiguousarray(param.detach().cpu().numpy()).astype("<f4", copy=False)

            # transpose tripwire. nn.Linear already stores weight as [out, in], which is
            # exactly the IR layout -- nothing here or in any kernel may transpose.
            if suffix == "weight":
                assert arr.shape == (layer.out_features, layer.in_features), arr.shape
            else:
                assert arr.shape == (layer.out_features,), arr.shape

            pad = align_up(offset, ALIGN) - offset
            blob += b"\x00" * pad
            offset += pad

            raw = arr.tobytes(order="C")
            manifest.append({
                "name": f"{name}.{suffix}",
                "dtype": "float32",
                "shape": list(arr.shape),
                "byte_offset": offset,
                "byte_len": len(raw),
            })
            blob += raw
            offset += len(raw)

    # tail pad so the total size is a multiple of ALIGN, which is what makes the C++
    # std::aligned_alloc call legal.
    blob += b"\x00" * (align_up(offset, ALIGN) - offset)

    assert all(e["byte_offset"] % ALIGN == 0 for e in manifest)
    assert all(e["byte_len"] == int(np.prod(e["shape"])) * 4 for e in manifest)
    assert len(blob) % ALIGN == 0
    return bytes(blob), manifest


def build_nodes(model):
    """Walk graph_layers() in order, threading the activation name forward."""
    nodes = []
    current = "input"
    features = None

    for name, layer in model.graph_layers():
        output = f"{name}.out"

        if isinstance(layer, nn.Linear):
            if features is None:
                features = layer.in_features
            assert layer.in_features == features, "feature chain broken at " + name
            node = {
                "op": "Linear",
                "name": name,
                "inputs": [current],
                "output": output,
                "weight": f"{name}.weight",
                "weight_shape": [layer.out_features, layer.in_features],
                "bias": f"{name}.bias",
                "bias_shape": [layer.out_features],
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }
            features = layer.out_features
        else:
            # ops without weights omit the weight keys entirely -- absent, not empty.
            node = {
                "op": "ReLU",
                "name": name,
                "inputs": [current],
                "output": output,
                "in_features": features,
                "out_features": features,
            }

        nodes.append(node)
        current = output

    return nodes


def render_ints(values):
    return "[" + ",".join(str(v) for v in values) + "]"


def render_graph_txt(graph):
    """Canonical textual rendering. The C++ format_graph() must reproduce this exactly.

    No column padding: alignment padding is where cross-language renderers diverge.
    """
    lines = []

    lines.append(
        f"graph ir_version={graph['ir_version']}"
        f" alignment={graph['alignment']}"
        f" weights_file={graph['weights_file']}"
        f" weights_total_bytes={graph['weights_total_bytes']}"
    )

    for gi in graph["graph_inputs"]:
        lines.append(
            f"input name={gi['name']} dtype={gi['dtype']}"
            f" shape={render_ints(gi['shape'])} features={gi['features']}"
        )

    for w in graph["weights"]:
        lines.append(
            f"weight name={w['name']} dtype={w['dtype']}"
            f" shape={render_ints(w['shape'])}"
            f" byte_offset={w['byte_offset']} byte_len={w['byte_len']}"
        )

    for i, n in enumerate(graph["nodes"]):
        parts = [
            f"node index={i}",
            f"op={n['op']}",
            f"name={n['name']}",
            f"inputs=[{','.join(n['inputs'])}]",
            f"output={n['output']}",
        ]
        if "weight" in n:
            parts += [
                f"weight={n['weight']}",
                f"weight_shape={render_ints(n['weight_shape'])}",
                f"bias={n['bias']}",
                f"bias_shape={render_ints(n['bias_shape'])}",
            ]
        parts += [
            f"in_features={n['in_features']}",
            f"out_features={n['out_features']}",
        ]
        lines.append(" ".join(parts))

    for name in graph["outputs"]:
        lines.append(f"output name={name}")

    return "\n".join(lines) + "\n"


def main():
    model = mlp_model.MLPClassifier()
    model.load_state_dict(
        torch.load("models/mlp_model.pt", weights_only=True, map_location="cpu")
    )
    model.eval()

    blob, manifest = build_weights(model)
    nodes = build_nodes(model)

    first_linear = next(l for _, l in model.graph_layers() if isinstance(l, nn.Linear))

    graph = {
        "ir_version": IR_VERSION,
        "weights_file": WEIGHTS_FILE,
        "alignment": ALIGN,
        "weights_total_bytes": len(blob),
        "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "graph_inputs": [{
            "name": "input",
            "dtype": "float32",
            # -1 is the runtime batch sentinel; `features` carries the real dimension so
            # no consumer has to interpret the sentinel to size a layer.
            "shape": [-1, first_linear.in_features],
            "features": first_linear.in_features,
        }],
        "weights": manifest,
        "nodes": nodes,
        "outputs": [nodes[-1]["output"]],
    }

    with open(WEIGHTS_FILE, "wb") as f:
        f.write(blob)

    with open("graph.json", "w") as f:
        json.dump(graph, f, indent=2)

    with open("graph.txt", "w", newline="\n") as f:
        f.write(render_graph_txt(graph))

    # standalone .npy per weight: validation fixtures only, never the runtime path.
    # they exist so a test can diff getWeight() against an independently-written file.
    os.makedirs("weights", exist_ok=True)
    for name, layer in model.graph_layers():
        if not isinstance(layer, nn.Linear):
            continue
        for suffix, param in (("weight", layer.weight), ("bias", layer.bias)):
            arr = np.ascontiguousarray(param.detach().cpu().numpy()).astype("<f4", copy=False)
            np.save(f"weights/{name}.{suffix}", arr)

    print(f"wrote {WEIGHTS_FILE} ({len(blob)} bytes, {len(manifest)} sections)")
    print(f"wrote graph.json ({len(nodes)} nodes)")
    print("wrote graph.txt")
    print(f"wrote weights/*.npy ({len(manifest)} fixtures)")


if __name__ == "__main__":
    main()
