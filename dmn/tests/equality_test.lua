require('..')

-- test tensors
local first_tensor = torch.rand(5)
local second_tensor = torch.rand(5)

local test_true = dmn.math_functions.equals(first_tensor, first_tensor:clone())
local test_false = dmn.math_functions.equals(first_tensor, torch.rand(5):zero())

assert(test_true)
assert(not test_false)

-- test tables
local test_true1 = dmn.math_functions.equals({first_tensor}, {first_tensor:clone()})
local test_false1 = dmn.math_functions.equals({first_tensor, first_tensor}, {first_tensor:clone()})
local test_false2 = dmn.math_functions.equals({first_tensor}, {first_tensor:clone(), first_tensor:clone()})

assert(test_true1)
assert(not test_false1)
assert(not test_false2)

local test_true2 = dmn.math_functions.equals({first_tensor, second_tensor}, {first_tensor:clone(), second_tensor:clone()})
local test_false3 = dmn.math_functions.equals({first_tensor, second_tensor}, {first_tensor:clone(), first_tensor:clone()})

assert(test_true2)
assert(not test_false3)