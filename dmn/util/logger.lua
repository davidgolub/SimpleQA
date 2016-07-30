--[[
Logger that logs for all the other loggers
]]

local Logger = torch.class('dmn.Logger')

function Logger:__init(config)
	self.loggers = {}
end

function Logger:add_logger(logger)
	assert(logger ~= nil, "Must specify logger to add")
	table.insert(self.loggers, logger)
end

function Logger:log(data)
    for i = 1, #self.loggers do 
    	cur_logger = self.loggers[i]
    	cur_logger:log(data)
    end
end

function Logger:print(data, ...)
    for i = 1, #self.loggers do 
    	cur_logger = self.loggers[i]
    	cur_logger:print(data, ...)
    end
end

function Logger:printf(data, ...)
	for i = 1, #self.loggers do 
    	cur_logger = self.loggers[i]
    	cur_logger:printf(data, ...)
    end
end

function Logger:header(s)
    for i = 1, #self.loggers do 
    	cur_logger = self.loggers[i]
    	cur_logger:header(s)
    end
end