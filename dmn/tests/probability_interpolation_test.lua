require('..')

-- Tests that probability interpolations work as expected

local input = torch.DoubleTensor{0.1,0.2,0.3,0.4}
local desired_class = torch.IntTensor{1}

local first_interpolation = dmn.math_functions.probability_interpolation(desired_class, 
	input,
	-1,
	1.0)

print(first_interpolation)
assert(first_interpolation[1] == 1)

local second_interpolation = dmn.math_functions.probability_interpolation(desired_class, 
	input,
	4,
	0.0)

print(second_interpolation)

local third_interpolation = dmn.math_functions.probability_interpolation(desired_class, 
	input,
	2,
	0.5)

print(third_interpolation)