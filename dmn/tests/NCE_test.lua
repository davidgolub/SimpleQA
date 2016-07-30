--[[
Gradient checks the noise-contrastive estimation criterion
]]

require('../../dmn')

model = nn.Sequential()
		:add(nn.Linear(50, 50))
		:add(nn.SoftMax())

criterion = dmn.NCECriterion()

local inputs = torch.rand(20, 50)
local sample_probs = torch.rand(20, 50):fill(0.001)

local params, grad_params = model:getParameters()
local currIndex = 0
local feval = function(x)
	grad_params:zero()
	local total_err = 0
	for i = 1, inputs:size(1) do
		local res = model:forward(inputs[i])
		
		-- accumulate total error
		curr_err = criterion:forward(res, sample_probs[i])
		total_err = total_err + curr_err

		input_err = criterion:backward(res, sample_probs[i])
		local input_grads = model:backward(inputs[i], input_err)
	end
	return total_err, grad_params
end

-- check gradients for lstm layer
diff, DC, DC_est = optim.checkgrad(feval, params, 1e-4)
print("Gradient error for document embed module network is")
print(diff)
assert(diff < 1e-5, "Gradient is greater than tolerance")

for i = 1, 100 do 
	optim.sgd(feval, params, {learningRate = 1e-1})
end

res = model:forward(inputs)

for i = 1, res:size(1) do
print(res[i][1])
print(res[i][{{2, 30}}])
end