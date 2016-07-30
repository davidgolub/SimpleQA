require('nn')

print("==== Testing temporal convolutions ====")

local input = torch.rand(5, 500)
local conv_network = nn.TemporalConvolution(500, 100, 2, 1)
local res = conv_network:forward(input)
print(res)

inp=5;  -- dimensionality of one sequence element
outp=1; -- number of derived features for one sequence element
kw=1;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

mlp=nn.TemporalConvolution(inp,outp,kw,dw)

x=torch.rand(7,inp) -- a sequence of 7 elements
print(mlp:forward(x))

y=torch.rand(15,inp) -- a sequence of 15 elements
print(mlp:forward(y))

print(mlp:forward(x) - mlp:forward(y))

