# import torch
# import torch.nn as nn
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
# import numpy as np
# import dgl
# import dgl.function as fn
#
# class PyG_conv(MessagePassing):
#     def __init__(
#             self,
#             in_channel,
#             out_channel,
#     ):
#         super(PyG_conv, self).__init__(aggr='add')
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.W = nn.Parameter(torch.ones((in_channel, out_channel)))
#         self.b = nn.Parameter(torch.ones(out_channel))
#
#     def forward(self,x, edge_index, edge_weight):
#         # edge_index,_=add_self_loops(edge_index,num_nodes=x[0])
#
#         x = torch.matmul(x, self.W)
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_weight=edge_weight)
#
#         # Your code here
#         pass
#         # End code here
#
#     def message(self,x_j, edge_weight):
#         return edge_weight.view(-1, 1) * x_j
#         # Your code here
#         pass
#         # End code here
#
#     def update(self, aggr_out, x):
#         return aggr_out + self.b
#
#
#
# # edge_index = torch.tensor([[0,1,1,2,2,4],[2,0,2,3,4,3]])
# # x = torch.ones((5, 8))
# # edge_weight = 2 * torch.ones(6)
# # conv = PyG_conv(8, 4)
# # output = conv(x, edge_index, edge_weight)
# # # assert np.allclose(output.detach().numpy(), [[17., 17., 17., 17.],
# # #                       [ 1.,  1.,  1.,  1.],
# # #                       [33., 33., 33., 33.],
# # #                       [33., 33., 33., 33.],
# # #                       [17., 17., 17., 17.]])
# # print(output)
# def message_func(edges):
#     return {'m': edges.src['h'] * edges.data['edge_weight']}
# class DGL_conv(nn.Module):
#   def __init__(self, in_channel, out_channel):
#     super(DGL_conv, self).__init__()
#     self.in_channel = in_channel
#     self.out_channel = out_channel
#     self.W = nn.Parameter(torch.ones(in_channel, out_channel))
#     self.b = nn.Parameter(torch.ones(out_channel))
#
#   def forward(self, g, h,edge_weight):
#     with g.local_scope():
#       g.ndata['h'] = h
#       g.edata['edge_weight'] = edge_weight
#       g.update_all(fn.u_mul_e('h', 'edge_weight', 'm'), fn.sum('m', 'h'))
#       h = g.ndata['h']
#       return torch.matmul(h, self.W) + self.b
#     # Your code here
#     pass
# src = torch.tensor([0, 1, 1, 2, 2, 4])
# dst = torch.tensor([2, 0, 2, 3, 4, 3])
# h = torch.ones((5, 8))
# g = dgl.graph((src, dst))
# edge_weight = 2 * torch.ones(6)
# conv = DGL_conv(8, 4)
# output = conv(g, h, edge_weight)
# print(output)
#     # import numpy as np
#     # assert np.allclose(output.detach().numpy(), [[17., 17., 17., 17.],
#     #                                              [1., 1., 1., 1.],
#     #                                              [33., 33., 33., 33.],
#     #                                              [33., 33., 33., 33.],
#     #                                              [17., 17., 17., 17.]])
#     # End code here
import torch
print(torch.__version__)