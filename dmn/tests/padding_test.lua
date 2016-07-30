require('..')

local paddedJoinTable = dmn.PaddedJoinTable(0)
local inputs = {torch.rand(5), torch.rand(2)}
local desired = torch.rand(2, 5)

local res = paddedJoinTable:forward(inputs)
local back = paddedJoinTable:backward(inputs, res)

print(res)
print(back)
--[[local params, grad_params = paddedJoinTable:getParameters()
local currIndex = 0
local loss_function = nn.MSECriterion()

local feval = function(x)
  grad_params:zero()
  local res = paddedJoinTable:forward(inputs)
  local err = loss_function:forward(res, desired)
  local err1 = loss_function:backward(res, desired)
  local input_errs = paddedJoinTable:backward(inputs, err1)

  currIndex = currIndex + 1
  print(currIndex, " of ", params:size())
  print(loss)
  return loss, grad_params
end

-- check gradients for lstm layer
diff, DC, DC_est = optim.checkgrad(feval, params, 1e-7)
print("Gradient error for dmn network is")
print(diff)
assert(diff < 1e-5, "Gradient is greater than tolerance")
]]
