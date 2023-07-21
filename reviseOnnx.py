import onnx
import onnx.helper as helper
import numpy as np
import torch
import torch.nn as nn

# model = onnx.load("attn_unet.onnx")
model = onnx.load("new.onnx")

# for node in model.graph.node:
#     print(node)

# for init in model.graph.initializer:
#     if init.name == "Conv.weight":
#         weight = np.frombuffer(init.raw_data, dtype=np.float32).reshape(*init.dims)
#         print(type(init))
#         print(weight.shape)
#         print(type(weight))

# 修改constant节点中的attribute
# for node in model.graph.node:
#     if(node.op_type == "Constant"):
#         if(node.output[0] == "onnx::Resize_353"):
#             print(f"{node.name}".center(15,'='))
#             #node.attribute[0].t.raw_data = np.array([999], dtype=np.float32).tobytes()
#             print(node)
#             print(np.frombuffer(node.attribute[0].t.raw_data, np.float32))

#覆盖Add节点
# new_add_node = helper.make_node("Add", ["onnx::Add_409", "onnx::Add_412"], 
#                                 ["input.116"], "Add_33_new")
# for node in model.graph.node:
#     if(node.name == "Add_33"):
#         node.CopyFrom(new_add_node)

#删除某个节点
# find_pre_node = lambda node:[x for x in model.graph.node if node.input[0] in x.output][0]
# find_next_node = lambda node:[x for x in model.graph.node if node.output[0] in x.input][0]
# remove_node = None
# for node in model.graph.node:
#     if node.name == "Relu_90":
#         pre = find_pre_node(node)
#         next = find_next_node(node)
#         next.input[0] = pre.output[0]
#         remove_node = node
# model.graph.node.remove(remove_node)

#修改网络输入、输出头
# input = model.graph.input[0]
# new_input = helper.make_tensor_value_info(input.name, 1, ["batch", 3, "w", "h"])
# input.CopyFrom(new_input)

# output = model.graph.output[0]
# new_output = helper.make_tensor_value_info(output.name, 1, ["batch", 4, "w", "h"])
# output.CopyFrom(new_output)


#依靠torch网络技巧，快速生成预处理onnx文件
# class Preprocess(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.mean = torch.rand(1, 1, 1, 3)
#         self.std = torch.rand(1, 1, 1, 3)
#     def forward(self, x):
#         x = x.float()
#         x = (x/255.0 - self.mean)/self.std
#         x = x.permute(0, 3, 1, 2)
#         return x

# preprocess_model = Preprocess()
# dummy = torch.zeros(1, 256, 256, 3)
# torch.onnx.export(
#     preprocess_model, 

#     # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
#     (dummy,), 

#     # 储存的文件路径
#     "preprocess.onnx", 

#     # 打印详细信息
#     verbose=False, 

#     input_names=["image_src"], 
#     output_names=["image_processed"], 


#     opset_version=11, 
# )

preprocess = onnx.load("preprocess.onnx")

preprocess.graph.input[0].name = "pre/" + preprocess.graph.input[0].name
#preprocess.graph.output[0].name = "pre/" + preprocess.graph.output[0].name

for node in preprocess.graph.node:
    node.name = "pre/" + node.name
    for i in range(len(node.input)):
        node.input[i] = "pre/" + node.input[i]
    for i in range(len(node.output)):
        node.output[i] = "pre/" + node.output[i]

for node in model.graph.node:
    if node.name == "Conv_3":
        #node.input[0] = preprocess.graph.output[0].name
        node.input[0] = "pre/" + preprocess.graph.output[0].name

for node in preprocess.graph.node:
    model.graph.node.append(node)

model.graph.input[0].CopyFrom(preprocess.graph.input[0])

onnx.save(model, "new5.onnx")
