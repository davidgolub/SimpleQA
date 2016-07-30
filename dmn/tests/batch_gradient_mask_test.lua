require('..')

tmp = torch.rand(5, 5, 2)
res1 = tmp[{{},{1}}]

tmp[{{2, 5},{1}}]:zero()

print(tmp)
print(tmp[{{1, 2},{1}}])
print(tmp[{{1, 2},{2}}])

local network = dmn.BatchMask()

local indices = torch.IntTensor(5, 2):fill(1)
indices[2]:fill(2)
indices[3]:fill(3)
local masked_input = network:forward(tmp, indices)
local input = network:backward(tmp, indices, torch.rand(5, 5, 2))

print(masked_input)
print(input)

tmp = torch.rand(5, 2)

local res1 = network:forward(tmp, indices)
local res2 = network:backward(tmp, indices, torch.rand(5, 2))

print(res1 - tmp)
print(res2)