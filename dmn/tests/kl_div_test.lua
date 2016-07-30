require('nn')

-- KL Divergence test. Turns out target needs to be a valid probability distribution, not a LOG of one.
input = torch.rand(5)

net = nn.Sequential() 
	:add(nn.Linear(5, 2))
	:add(nn.LogSoftMax())

probabilizer = nn.SoftMax()
log_probabilizzer = nn.LogSoftMax()

criterion = nn.DistKLDivCriterion()

res = net:forward(input)
desired = probabilizer:forward(torch.rand(2))
log_desired = log_probabilizzer:forward(torch.rand(2))

err = criterion:forward(res, desired)
err1 = criterion:forward(res, log_desired)

print(err)
print(err1)
