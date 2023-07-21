import onnx
import onnx.helper as helper
import numpy as np
import torch
import torch.nn as nn

def make_resize_scale(scale_name):
    scale = helper.make_tensor(
        name=scale_name,
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[4],
        vals=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32).tobytes(),
        raw=True
    )
    return scale
    
if __name__ == "__main__":
    model = onnx.load("workspace/attn_unet2.onnx")

    # 1.处理Resize_44
    scale_1 = make_resize_scale("scale_1")
    model.graph.initializer.append(scale_1)

    r1_new = helper.make_node(
        op_type = "Resize",
        name = "Resize_44_new",
        inputs=["onnx::Resize_299", "onnx::Resize_303", "scale_1"],
        outputs=["input.144"],
        coordinate_transformation_mode = "asymmetric",
        cubic_coeff_a = -0.75,
        mode = "nearest",
        nearest_mode = "floor"
    )
    
    for node in model.graph.node:
        if node.op_type == "Resize" and node.name == "Resize_44":
            r1 = node
            r1.CopyFrom(r1_new)
            break
    for node in model.graph.node:
        if node.op_type == "Identity" and node.name == "Identity_2":
            model.graph.node.remove(node)
            break
    
    # 2.处理Resize_60
    scale_2 = make_resize_scale("scale_2")
    model.graph.initializer.append(scale_2)

    r2_new = helper.make_node(
        op_type = "Resize",
        name = "Resize_60_new",
        inputs=["onnx::Resize_324", "onnx::Resize_328", "scale_2"],
        outputs=["input.192"],
        coordinate_transformation_mode = "asymmetric",
        cubic_coeff_a = -0.75,
        mode = "nearest",
        nearest_mode = "floor"
    )
    
    for node in model.graph.node:
        if node.op_type == "Resize" and node.name == "Resize_60":
            r2 = node
            r2.CopyFrom(r2_new)
    for node in model.graph.node:
        if node.op_type == "Identity" and node.name == "Identity_1":
            model.graph.node.remove(node)
            break
    
    
    # 3.处理Resize_76
    scale_3 = make_resize_scale("scale_3")
    model.graph.initializer.append(scale_3)

    r3_new = helper.make_node(
        op_type = "Resize",
        name = "Resize_76_new",
        inputs=["onnx::Resize_349", "onnx::Resize_353", "scale_3"],
        outputs=["input.240"],
        coordinate_transformation_mode = "asymmetric",
        cubic_coeff_a = -0.75,
        mode = "nearest",
        nearest_mode = "floor"
    )
    
    for node in model.graph.node:
        if node.op_type == "Resize" and node.name == "Resize_76":
            r3 = node
            r3.CopyFrom(r3_new)
    for node in model.graph.node:
        if node.op_type == "Identity" and node.name == "Identity_0":
            model.graph.node.remove(node)
            break
    
    
    # model.graph.node.remove(r1)
    # model.graph.node.append(r1_new)
    onnx.save_model(model, "workspace/attn_unet_fresh2.onnx")
            
        