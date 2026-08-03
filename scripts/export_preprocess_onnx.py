from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    constants = [
        numpy_helper.from_array(np.array([], dtype=np.float32), "roi"),
        numpy_helper.from_array(np.array([], dtype=np.float32), "scales"),
        numpy_helper.from_array(np.array([1, 3, 224, 288], dtype=np.int64), "sizes"),
        numpy_helper.from_array(np.array(255.0, dtype=np.float32), "scale_255"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "mean"),
        numpy_helper.from_array(np.array([1, 14, 16, 18, 16, 3], dtype=np.int64), "shape_grid"),
        numpy_helper.from_array(np.array([1, 252, 768], dtype=np.int64), "shape_patches"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 4, 0], dtype=np.int64), "pads"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
    ]
    nodes = [
        helper.make_node(
            "Resize", ["rgb_chw_0_255", "roi", "scales", "sizes"], ["resized"],
            mode="linear", coordinate_transformation_mode="half_pixel",
            antialias=1, keep_aspect_ratio_policy="stretch",
        ),
        helper.make_node("Cast", ["resized"], ["resized_float"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["resized_float", "scale_255"], ["rescaled"]),
        helper.make_node("Sub", ["rescaled", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "mean"], ["normalized"]),
        helper.make_node("Transpose", ["normalized"], ["channels_last"], perm=[0, 2, 3, 1]),
        helper.make_node("Reshape", ["channels_last", "shape_grid"], ["grid"]),
        helper.make_node("Transpose", ["grid"], ["ordered"], perm=[0, 1, 3, 2, 4, 5]),
        helper.make_node("Reshape", ["ordered", "shape_patches"], ["patches"]),
        helper.make_node("Pad", ["patches", "pads", "zero"], ["pixel_values"], mode="constant"),
    ]
    graph = helper.make_graph(
        nodes, "siglip2_uvc_preprocess",
        [helper.make_tensor_value_info(
            "rgb_chw_0_255", TensorProto.UINT8, [1, 3, 768, 1024]
        )],
        [helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT, [1, 256, 768]
        )],
        initializer=constants,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)],
        producer_name="nailfold-report",
    )
    model.ir_version = 10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    onnx.checker.check_model(str(args.output))


if __name__ == "__main__":
    main()
