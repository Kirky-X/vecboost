#!/usr/bin/env python3
"""生成 KV 缓存 ScatterND 合并图：28 层 × (k,v)，显存内把 delta 写入缓存窗口。
输入：cache_k_i/cache_v_i [B,512,8,128]、delta_k_i/delta_v_i [B,seq,8,128]、indices [N,2]
输出：new_k_i/new_v_i [B,512,8,128]，其中 [b,pos]∈indices 的窗口被 delta 覆盖。
"""
import onnx
from onnx import helper, TensorProto

LAYERS = 28
KV_SHAPE = ["B", 512, 8, 128]
DELTA_SHAPE = ["B", "seq", 8, 128]

inputs, outputs, nodes = [], [], []
init_z2 = helper.make_tensor("shape_z2", TensorProto.INT64, [1], [2])
init_flat = helper.make_tensor("flat_shape", TensorProto.INT64, [3], [-1, 8, 128])
inits = [init_z2, init_flat]
for l in range(LAYERS):
    for kind in ("k", "v"):
        data_in = helper.make_tensor_value_info(f"cache_{kind}_{l}", TensorProto.FLOAT, KV_SHAPE)
        delta_in = helper.make_tensor_value_info(f"delta_{kind}_{l}", TensorProto.FLOAT, DELTA_SHAPE)
        out = helper.make_tensor_value_info(f"new_{kind}_{l}", TensorProto.FLOAT, KV_SHAPE)
        inputs += [data_in, delta_in]
        outputs.append(out)
        # ScatterND 要求 updates 秩 = data 秩 - indices 末维 + indices 秩 - 1 = 3，
        # delta [B,seq,8,128] 先 Reshape 成 [B*seq, 8, 128]
        nodes.append(helper.make_node(
            "Reshape",
            inputs=[f"delta_{kind}_{l}", "flat_shape"],
            outputs=[f"delta_flat_{kind}_{l}"],
            name=f"reshape_{kind}_{l}",
        ))
        nodes.append(helper.make_node(
            "ScatterND",
            inputs=[f"cache_{kind}_{l}", "indices", f"delta_flat_{kind}_{l}"],
            outputs=[f"new_{kind}_{l}"],
            name=f"scatter_{kind}_{l}",
        ))

indices_in = helper.make_tensor_value_info("indices", TensorProto.INT64, ["N", 2])
inputs.append(indices_in)

graph = helper.make_graph(
    nodes, "kv_scatter",
    inputs, outputs, initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "assets/models/scatter.onnx")
print("scatter.onnx 生成完成")
