--[[
Helper functions
]]

local Logger = torch.class('dmn.PrintLogger')

function Logger:__init(config)
end

function Logger:log(data)
    print(data)
end

function Logger:print(data, ...)
    print(data, ...)
end

function Logger:printf(data, ...)
	utils.printf(data, ...)
end

function Logger:header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end