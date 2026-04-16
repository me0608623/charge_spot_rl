import torch
from icecream import ic
t1, t2, t3, t4, t5, t6, t7, t8 = 1, 2, 3, 4, 5, 6, 7, 8

a = torch.tensor([
    [
        [[t1,t1,t1,t1,t1,t1],[t1,t1,t1,t1,t1,t1]],
        [[t1,t1,t1,t1,t1,t1],[t1,t1,t1,t1,t1,t1]]
    ],
    [
        [[t2,t2,t2,t2,t2,t2],[t2,t2,t2,t2,t2,t2]],
        [[t2,t2,t2,t2,t2,t2],[t2,t2,t2,t2,t2,t2]]
    ],
    [
        [[t3,t3,t3,t3,t3,t3],[t3,t3,t3,t3,t3,t3]],
        [[t3,t3,t3,t3,t3,t3],[t3,t3,t3,t3,t3,t3]]
    ],
    [
        [[t4,t4,t4,t4,t4,t4],[t4,t4,t4,t4,t4,t4]],
        [[t4,t4,t4,t4,t4,t4],[t4,t4,t4,t4,t4,t4]]
    ], 
    [
        [[t5,t5,t5,t5,t5,t5],[t5,t5,t5,t5,t5,t5]],
        [[t5,t5,t5,t5,t5,t5],[t5,t5,t5,t5,t5,t5]]
    ], 
    [
        [[t6,t6,t6,t6,t6,t6],[t6,t6,t6,t6,t6,t6]],
        [[t6,t6,t6,t6,t6,t6],[t6,t6,t6,t6,t6,t6]]
    ], 
    [
        [[t7,t7,t7,t7,t7,t7],[t7,t7,t7,t7,t7,t7]],
        [[t7,t7,t7,t7,t7,t7],[t7,t7,t7,t7,t7,t7]]
    ], 
    [
        [[t8,t8,t8,t8,t8,t8],[t8,t8,t8,t8,t8,t8]],
        [[t8,t8,t8,t8,t8,t8],[t8,t8,t8,t8,t8,t8]]
    ], 
])
seq_len = 4 
b = a.reshape(-1, seq_len, a.size(1),a.size(2), a.size(3))
seq_date = b.permute(1, 0, 2, 3, 4)
rin = seq_date.reshape(seq_len, -1, a.size(3))
ic(rin)

seq_date = rin.reshape(seq_len, -1, a.size(1),a.size(2), a.size(3))
seq_date = seq_date.permute(1, 0, 2, 3, 4)
output = seq_date.reshape(8, 2, 2,6)
ic(output)
