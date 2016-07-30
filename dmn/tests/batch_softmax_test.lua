require('..')

local network = nn.Sequential()
					:add(dmn.BatchReshape())
					:add(nn.Linear(500, 20))
					:add(nn.LogSoftMax())

local inputs = torch.rand(3, 5, 500)
local results = network:forward(inputs)

print(results)
local labels = torch.IntTensor{{1, 2, 3, 4, 5}, 
{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}

local reshaped_labels = labels:view(15)
print(reshaped_labels)
print(results)
local criterion = nn.ClassNLLCriterion()

local err = criterion:forward(results, reshaped_labels)
local input_err = criterion:backward(results, reshaped_labels)
local input_grads = network:backward(inputs, input_err)

print(input_grads)

for i = 1, inputs:size(2) do
	local cur_input = inputs[{{}, i}]
	local cur_label = labels[{{}, i}]

	local single_res = network:forward(cur_input)
	local single_err = criterion:forward(single_res, cur_label)

	local cur_input_err = criterion:backward(single_res, cur_label)
	local cur_input_grads = network:backward(cur_input, cur_input_err)

	local grad_diff = cur_input_grads - input_grads[{{}, i}]
	print(cur_input_grads:cdiv(input_grads[{{}, i}]))
	print(torch.abs(grad_diff):sum())
end