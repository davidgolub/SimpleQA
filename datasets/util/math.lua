--[[

  Various utility functions on tables

--]]

-- onelined version ;)
--    getPath=function(str,sep)sep=sep or'/'return str:match("(.*"..sep..")")end
function get_dir(str,sep)
    sep=sep or'/'
    return str:match("(.*"..sep..")")
end

-- makes a directory if not found in path
function make_dir(path)
  local base_dir = get_dir(path)
  if lfs.attributes(base_dir) == nil then
    print("Directory not found for " .. path .. ", making new directory at " .. base_dir)
    lfs.mkdir(base_dir)
  end
end

function sleep(s)
  local ntime = os.time() + s
  repeat until os.time() > ntime
end

function trim(string)
  return (string:gsub("^%s*(.-)%s*$", "%1"))
end

-- Make a deep copy of a table
function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- Check type of input
function check_type(input, desired_type)
  assert(input ~= nil, "Must specify input type for check_type function")
  assert(desired_type ~= nil, "Must specify desired type for check_type function")

  local input_type = torch.typename(input) or "NULL"
  assert(input_type == desired_type, "input has type " .. input_type .. " but desired is " .. desired_type)
end

-- returns a new zero unit (of cuda or cpu mode)
function new_zero_unit(gpu_mode, mem_dim)
  return gpu_mode and torch.zeros(mem_dim):cuda()
         or torch.zeros(mem_dim)
end

function check_valid_gpu_inputs(inputs, gpu_mode)
assert(inputs ~= nil, "Must specify inputs to forward")
assert(gpu_mode ~= nil, "Must specify whether to use gpu mode or not")

local corr_type = gpu_mode and 'torch.CudaTensor' or 'torch.DoubleTensor'
  check_type(inputs, corr_type)
end

function new_hidden_activations_lstm(gpu_mode, mem_dim, num_layers) 
  if num_layers == 1 then 
    return {new_zero_unit(gpu_mode, mem_dim), new_zero_unit(gpu_mode, mem_dim)}
  else 
    local modules = {{},{}}
    for i = 1, num_layers do
      table.insert(modules[1], new_zero_unit(gpu_mode, mem_dim))
      table.insert(modules[2], new_zero_unit(gpu_mode, mem_dim))
    end
    return modules
  end
end

-- Enable dropouts
function enable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:training()
      end
   end
end

-- Disable dropouts
function disable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:evaluate()
      end
   end
end

-- adds modules into parallel network from module list
-- requires parallel_net is of type nn.parallel
-- requires module_list is an array of modules that is not null
-- modifies: parallel_net by adding modules into parallel net
function add_modules(parallel_net, module_list)
  assert(parallel_net ~= nil, "parallel net is null")
  assert(module_list ~= nil, "modules you're trying to add are null")

  for i = 1, #module_list do
    curr_module = module_list[i]
    parallel_net:add(curr_module)
  end
end

-- adds modules into parallel network from module list
-- requires parallel_net is of type nn.parallel
-- requires module_list is an array of modules that is not null
-- modifies: parallel_net by adding modules into parallel net
function insert_modules_to_table(curr_table, mod_list)
  assert(curr_table ~= nil)
  assert(mod_list ~= nil, "Module list must not be null")

  for i = 1, #mod_list do
    curr_module = mod_list[i]
    table.insert(curr_table, curr_module)
  end
end

-- Convert 1-d torch tensor to lua table
function tensor_to_array(t1)
  -- This assumes `t1` is a 2-dimensional tensor!
  local t2 = {}
  for i=1,t1:size(1) do
    t2[i] = t1[i]
  end
  return t2
end



