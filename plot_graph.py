from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

from unit_tests import transformer_onnx

pydot_graph = GetPydotGraph(
    transformer_onnx.graph,
    name=transformer_onnx.graph.name,
    rankdir="TB",
    node_producer=GetOpNodeProducer("docstring"),
)
pydot_graph.write_png("graph.png")
