"""Writers of the files the weight preparation and GGUF metadata tests read:
safetensors checkpoints, GGUF files and prepared weight files."""

import json
import struct

# model::kWeightFileAlignment (runtime/model/WeightLayout.hpp): every section
# of a prepared weight file starts at a multiple of it.
WEIGHT_FILE_ALIGNMENT = 16384
# The alignment of GGUF tensor data without a general.alignment key.
GGUF_ALIGNMENT = 32
# The GGUF value types (gguf_type) of the values write_gguf encodes.
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def weight_file(magic, layer, kind, sections):
    """A prepared weight file: its 16-byte header (the eight-byte magic, then
    layer and type), then each section at the next alignment boundary."""
    data = bytearray(struct.pack("<8sII", magic.encode(), layer, kind))
    data.extend(bytes(WEIGHT_FILE_ALIGNMENT - len(data)))
    for section in sections:
        data.extend(section)
        data.extend(bytes(-len(data) % WEIGHT_FILE_ALIGNMENT))
    return bytes(data)


def safetensors_bytes(header, payload):
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


def write_safetensors(path, tensors):
    """Writes tensors, name: (shape, dtype, data), in order."""
    header, payload = {}, bytearray()
    for name, (shape, dtype, data) in tensors.items():
        header[name] = {
            "shape": shape,
            "dtype": dtype,
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    path.write_bytes(safetensors_bytes(header, payload))


def read_safetensors(path):
    """The header and the tensor data of a safetensors file."""
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8 : 8 + length]), data[8 + length :]


def _gguf_string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _gguf_value(value):
    """The GGUF value type and encoding of a string, bool, int (uint32),
    float (float32) or list of one of them."""
    if isinstance(value, str):
        return GGUF_TYPE_STRING, _gguf_string(value)
    if isinstance(value, bool):
        return GGUF_TYPE_BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        return GGUF_TYPE_UINT32, struct.pack("<I", value)
    if isinstance(value, float):
        return GGUF_TYPE_FLOAT32, struct.pack("<f", value)
    if isinstance(value, list):
        kind = _gguf_value(value[0])[0] if value else GGUF_TYPE_UINT32
        return GGUF_TYPE_ARRAY, struct.pack("<IQ", kind, len(value)) + b"".join(
            _gguf_value(x)[1] for x in value
        )
    raise AssertionError(value)


def write_gguf(path, metadata, tensors=()):
    """Writes a version 3 GGUF of the metadata and the tensors, each (name,
    shape, GGML type, data), the data aligned after the tensor table. A file
    whose tensors have no data ends at the tensor table."""
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        kind, encoded = _gguf_value(value)
        header.extend(_gguf_string(key) + struct.pack("<I", kind) + encoded)
    data = bytearray()
    for name, shape, kind, raw in tensors:
        header.extend(_gguf_string(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", kind, len(data)))
        data.extend(raw)
        data.extend(bytes(-len(data) % GGUF_ALIGNMENT))
    if data:
        header.extend(bytes(-len(header) % GGUF_ALIGNMENT))
    path.write_bytes(header + data)
    return path
