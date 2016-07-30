require('..')

local tensor = torch.rand(8, 2)

local indices = {dmn.constants.TRAIN_INDEX, 
dmn.constants.TRAIN_INDEX, 
dmn.constants.VAL_INDEX, 
dmn.constants.TEST_INDEX,
dmn.constants.TEST_INDEX,
dmn.constants.TRAIN_INDEX,
dmn.constants.VAL_INDEX,
dmn.constants.TEST_INDEX}

local train_tensor, val_tensor, test_tensor = 
	dmn.functions.partition_tensor(tensor, indices)

print(train_tensor)
print(val_tensor)
print(test_tensor)

print(tensor)