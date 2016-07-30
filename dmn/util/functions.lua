--[[
  Helper functions
]]

local functions = torch.class('dmn.functions')

-- sets gpu mode on an array of items
function functions.set_gpu_mode(items)
	assert(items ~= nil, "Must specify items to set gpu mode on")
	for i = 1, #items do
		print("Setting gpu mode on ", items[i])
		items[i]:set_gpu_mode()
	end
end

function functions.convert_to_gpu(tab, gpu_mode)
  assert(tab ~= nil, "Must specify whether table to convert")
  assert(gpu_mode ~= nil, "Must specify whether to convert to gpu mode or not")
  -- move optim state to double as well
  for k,v in pairs(tab) do 
    if gpu_mode then 
      if torch.typename(v) == 'torch.CudaTensor' then 
        tab[k] = v:double()
      end
    else 
      if torch.typename(v) == 'torch.DoubleTensor' then 
        tab[k] = v:cuda()
      end
    end 
  end

end

-- Strip accents from a string
function functions.strip_accents( str )
  if functions.tableAccents == nil then 
      tableAccents = {}
      tableAccents["à"] = "a"
      tableAccents["á"] = "a"
      tableAccents["â"] = "a"
      tableAccents["ã"] = "a"
      tableAccents["ä"] = "a"
      tableAccents["ç"] = "c"
      tableAccents["è"] = "e"
      tableAccents["é"] = "e"
      tableAccents["ê"] = "e"
      tableAccents["ë"] = "e"
      tableAccents["ì"] = "i"
      tableAccents["í"] = "i"
      tableAccents["î"] = "i"
      tableAccents["ï"] = "i"
      tableAccents["ñ"] = "n"
      tableAccents["ò"] = "o"
      tableAccents["ó"] = "o"
      tableAccents["ô"] = "o"
      tableAccents["õ"] = "o"
      tableAccents["ö"] = "o"
      tableAccents["ù"] = "u"
      tableAccents["ú"] = "u"
      tableAccents["û"] = "u"
      tableAccents["ü"] = "u"
      tableAccents["ý"] = "y"
      tableAccents["ÿ"] = "y"
      tableAccents["À"] = "A"
      tableAccents["Á"] = "A"
      tableAccents["Â"] = "A"
      tableAccents["Ã"] = "A"
      tableAccents["Ä"] = "A"
      tableAccents["Ç"] = "C"
      tableAccents["È"] = "E"
      tableAccents["É"] = "E"
      tableAccents["Ê"] = "E"
      tableAccents["Ë"] = "E"
      tableAccents["Ì"] = "I"
      tableAccents["Í"] = "I"
      tableAccents["Î"] = "I"
      tableAccents["Ï"] = "I"
      tableAccents["Ñ"] = "N"
      tableAccents["Ò"] = "O"
      tableAccents["Ó"] = "O"
      tableAccents["Ô"] = "O"
      tableAccents["Õ"] = "O"
      tableAccents["Ö"] = "O"
      tableAccents["Ù"] = "U"
      tableAccents["Ú"] = "U"
      tableAccents["Û"] = "U"
      tableAccents["Ü"] = "U"
      tableAccents["Ý"] = "Y"

      functions.tableAccents = tableAccents
  end
        
    local normalizedString = ""
 
    for strChar in string.gfind(str, "([%z\1-\127\194-\244][\128-\191]*)") do
        if functions.tableAccents[strChar] ~= nil then
            normalizedString = normalizedString .. " " --tableAccents[strChar]
        else
            normalizedString = normalizedString .. strChar
        end
    end
        
  return normalizedString
 
end

function functions.string_trim(s)
  assert(s ~= nil, "Must specify string to trim")
  return s:gsub("^%s*(.-)%s*$", "%1")
end

function functions.string_starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function functions.module_exists(module_name)
  assert(module_name ~= nil, "Must specify module to load")
  local has_module, mod = pcall(require,module_name)
  return has_module
end

function functions.strip_nonascii(str)
  local s = ""
  for i = 1, str:len() do
    if str:byte(i) >= 32 and str:byte(i) <= 126 then
        s = s .. str:sub(i,i)
    end
  end
  return s
end

-- creates a dictionary from a table object, converts each object to a key
function functions.create_dictionary(original_table)
  assert(original_table ~= nil, "Must specify original table to create keys for")
  local new_table = {}
  for i = 1, #original_table do
    local key = original_table[i]
    new_table[key] = i
  end
  return new_table
end

-- gets all keys from table
function functions.keys(cur_table, copy_table)
  assert(cur_table ~= nil, "Must specify table to extract keys from")
  local new_table = {}
  local new_copy_table = {}
  for k, v in pairs(cur_table) do
    table.insert(new_table, k)
    if copy_table ~= nil then
      table.insert(new_copy_table, copy_table[v])
    end
  end

  return new_table, new_copy_table
end

-- unique values from original table. Optionally takes a "copy table" to get unique vals from that too
function functions.unique_values(original_table, copy_table)
  local dictionary = functions.create_dictionary(original_table)
  local keys, new_copy_table = functions.keys(dictionary, copy_table)
  return keys, new_copy_table
end

-- Counts number of unique values in table. Divides by table size
function functions.count_values(original_table)
  local counts = {}
  for i = 1, #original_table do
    local cur_val = original_table[i]
    if counts[cur_val] == nil then 
      counts[cur_val] = 1
    else
      counts[cur_val] = counts[cur_val] + 1
    end
  end
  for k, v in pairs(counts) do
    counts[k] = v / #original_table
  end
  return counts
end

-- Converts table of strings into a table of ngrams, 
function functions.ngrams(original_table, n)
  assert(original_table ~= nil, "Must specify table to ngram")
  assert(n ~= nil, "Must specify number of strings to ngram")

  local new_table = {}
  for i = 1, #original_table - n + 1 do
    new_str = ''
    for j = 0, n - 2 do
      cur_str = original_table[i + j]
      new_str = new_str .. cur_str .. ' '
    end
      new_str = new_str .. original_table[i + n - 1]
    table.insert(new_table, new_str)
  end 

  return new_table
end

function functions.collect_garbage()
  collectgarbage()
  dmn.logger:print("Collecting garbage, current free memory is " .. collectgarbage("count")*1024)
end

-- adds modules into parallel network from module list
-- requires parallel_net is of type nn.parallel
-- requires module_list is an array of modules that is not null
-- modifies: parallel_net by adding modules into parallel net
function functions.add_modules(parallel_net, module_list)
  assert(parallel_net ~= nil, "parallel net is null")
  assert(module_list ~= nil, "modules you're trying to add are null")
  for i = 1, #module_list do
    curr_module = module_list[i]
    parallel_net:add(curr_module)
  end
end

-- Enable dropouts
function functions.enable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:training()
      end
   end
end

-- Disable dropouts
function functions.disable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        print("Evaluate model")
        m:evaluate()
      end
   end
end

-- for unescaping urls
function functions.hex_to_char(x)
  assert(x ~= nil, "Must specify hex to unescape")
    return string.char(tonumber(x, 16))
end

-- for unescaping urls
function functions.unescape(url)
  assert(url ~= nil, "Must specify url to unescape")
  return url:gsub("%%(%x%x)", functions.hex_to_char)
end

-- for unescaping html
function functions.html_unescape(str)
  str = string.gsub( str, '&lt;', '<' )
  str = string.gsub( str, '&gt;', '>' )
  str = string.gsub( str, '&quot;', '"' )
  str = string.gsub( str, '&apos;', "'" )
  str = string.gsub( str, '&#(%d+);', function(n) return string.char(n) end )
  str = string.gsub( str, '&#x(%d+);', function(n) return string.char(tonumber(n,16)) end )
  str = string.gsub( str, '&amp;', '&' ) -- Be sure to do this after all others
  return str
end

-- creates train/val/test split indexes for data
function functions.train_val_test_split(num_items, ratio)
  assert(num_items > 0, "Number of items to train/val/split must be greater than zero")
  assert(ratio ~= nil, "Ratio must not be null")
  assert(ratio >= 0 and ratio <= 1, 
    "Train/(test + val) ratio must be greater than zero and less than one")

  local splits = {}
  local train_threshold = ratio
  local val_threshold = ratio + (1 - ratio) / 2


  assert(val_threshold > train_threshold, "Will get zero val samples")

  for i = 1, num_items do 
    local sample = torch.uniform()
    local index 

    if sample < train_threshold then 
      index = dmn.constants.TRAIN_INDEX
    elseif sample < val_threshold then
      index = dmn.constants.VAL_INDEX
    else 
      index = dmn.constants.TEST_INDEX
    end
    table.insert(splits, index) 
  end
  
  return splits
end

-- Sorts tables by first value
-- first_entry, second_entry are tables
function functions.min_sort_function(first_table, second_table)
    return first_table[1] < second_table[1]
end


-- Sorts tables by first value
-- first_entry, second_entry are tables
function functions.max_sort_function(first_table, second_table)
    return first_table[1] > second_table[1]
end

-- Argmax: Returns max values of tensor along last dimension given
function functions.argmax(v)
  local dim_for_max = v:dim()
  local vals, indices = torch.max(v, dim_for_max)
  return torch.squeeze(indices)
end

-- TopkArgmax returns top k indices, values from list
-- First value is comparator, second is values
function functions.topk(list, k)
  tmp_list = {}
  for i = 1, #list do
    table.insert(tmp_list, list[i])
  end
  table.sort(tmp_list, functions.max_sort_function)

  max_entries = {}
  for i = 1, k do
    table.insert(max_entries, tmp_list[i])
  end

  return max_entries
end

-- TopkArgmax returns top k indices, values from list
function functions.topkargmax(list, k)
  local cloned_list = list:clone()
  max_indices = {}
  for i = 1, k do
    local vals, indices = torch.max(cloned_list, 1)
    local best_index = indices[1]
    cloned_list[best_index] = -1000
    table.insert(max_indices, best_index)
  end
  return max_indices
end

-- partitions data based on train/test/split indeces
function functions.partition_data(data, indeces)
  assert(data ~= nil, "Must specify data to partition")
  assert(indeces ~= nil, "Must specify indeces to partition data with")
  assert(#data == #indeces, "Data size must equal indeces size")

  local train_data = {}
  local val_data = {}
  local test_data = {}

  for i = 1, #data do
    local data_point = data[i]
    local cur_index = indeces[i]
    if cur_index == dmn.constants.TRAIN_INDEX then 
      table.insert(train_data, data_point)
    elseif cur_index == dmn.constants.VAL_INDEX then
      table.insert(val_data, data_point)
    else
      table.insert(test_data, data_point)
    end
  end

  return train_data, val_data, test_data
end


-- partitions tensor data based on train/test/split indeces, returns train, val test tensors
function functions.partition_tensor(data, indeces)
  assert(data ~= nil, "Must specify data to partition")
  assert(indeces ~= nil, "Must specify indeces to partition data with")
  assert(data:size(1) == #indeces, "Data size must equal indeces size " .. data:size(1) .. " " .. #indeces)

  local num_train_data = 0
  local num_val_data = 0
  local num_test_data = 0

  -- First count how many things there are
  for i = 1, data:size(1) do
    if i % 10000 == 0 then
      dmn.logger:print("Tensor partition: on index " .. i .. " of " .. data:size(1))
    end

    local data_point = data[i]
    local cur_index = indeces[i]
    if cur_index == dmn.constants.TRAIN_INDEX then 
      num_train_data = num_train_data + 1
    elseif cur_index == dmn.constants.VAL_INDEX then
      num_val_data = num_val_data + 1
    else
      num_test_data = num_test_data + 1
    end
  end

  -- Then transfer everything into tensors
  train_tensor = torch.DoubleTensor(num_train_data, data:size(2))
  val_tensor = torch.DoubleTensor(num_val_data, data:size(2))
  test_tensor = torch.DoubleTensor(num_test_data, data:size(2))

  local cur_train_index = 1
  local cur_val_index = 1
  local cur_test_index = 1

   -- Now that we know how many things there are, insert them in
  for i = 1, data:size(1) do
    if i % 10000 == 0 then
      dmn.logger:print("Tensor partition: on index " .. i .. " of " .. data:size(1))
    end

    local data_point = data[i]
    local cur_index = indeces[i]
    if cur_index == dmn.constants.TRAIN_INDEX then 
      train_tensor[cur_train_index] = data_point
      cur_train_index = cur_train_index + 1
    elseif cur_index == dmn.constants.VAL_INDEX then
      val_tensor[cur_val_index] = data_point
      cur_val_index = cur_val_index + 1
    else
      test_tensor[cur_test_index] = data_point
      cur_test_index = cur_test_index + 1
    end
  end

  dmn.functions.collect_garbage()
  local total_size = val_tensor:size(1) + train_tensor:size(1) + test_tensor:size(1)
  local original_size = data:size(1)
  local msg = "Partitioned tensor size must equal original "
  assert(total_size == original_size, msg .. total_size .. " " .. original_size)
  return train_tensor, val_tensor, test_tensor
end

-- for splitting a string
function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

-- for trimming a string
function functions.trim(s)
  return s:gsub("^%s+", ""):gsub("%s+$", "")
end

-- for generating a unique id
function functions.uuid()
    local random = math.random
    local template ='xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'
    return string.gsub(template, '[xy]', function (c)
        local v = (c == 'x') and random(0, 0xf) or random(8, 0xb)
        return string.format('%x', v)
    end)
end

-- sets gpu mode on an array of items
function functions.set_cpu_mode(items)
	assert(items ~= nil, "Must specify items to set cpu mode on")
	for i = 1, #items do
		print("Setting cpu mode on ", items[i])
		items[i]:set_cpu_mode()
	end
end

-- enables dropouts on an array of items
function functions.enable_dropouts(items)
	assert(items ~= nil, "Must specify items to enable dropouts on")
	for i = 1, #items do
	  	--print("Enabling dropouts on ", items[i])
	    items[i]:enable_dropouts()
	end
end

-- disables dropouts on a narray of items
function functions.disable_dropouts(items)
	assert(items ~= nil, "Must specify items to enable dropouts on")
	for i = 1, #items do
	  	--print("Disabling dropouts on ", items[i])
	    items[i]:disable_dropouts()
	end
end

-- Concatenates two tables together
function functions.table_concat(t1, t2)
	assert(t1 ~= nil, "Must specify first table to concat")
	assert(t2 ~= nil, "Must specify second table to concat")

    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end
  

function functions.make_dir(path)
  local base_dir = get_dir(path)
  if lfs.attributes(base_dir) == nil then
    dmn.logger:print("Directory not found for " .. path .. ", making new directory at " .. base_dir)
    lfs.mkdir(base_dir)
  end
end

-- Make a deep copy of a table
function functions.deepcopy(orig)
    assert(orig ~= nil, "Must specify whether to copy previous outputs or not")
    file = torch.MemoryFile() -- creates a file in memory
    file:writeObject(orig) -- writes the object into file
    file:seek(1) -- comes back at the beginning of the file
    local objectClone = file:readObject() -- gets a clone of object
    return objectClone
end

-- exports string
local function exportstring( s )
return string.format("%q", s)
end

--// Saves a table
function functions.table_save(  tbl,filename )
local charS,charE = "   ","\n"
local file,err = io.open( filename, "wb" )
if err then return err end

-- initiate variables for save procedure
local tables,lookup = { tbl },{ [tbl] = 1 }
file:write( "return {"..charE )

for idx,t in ipairs( tables ) do
 file:write( "-- Table: {"..idx.."}"..charE )
 file:write( "{"..charE )
 local thandled = {}

 for i,v in ipairs( t ) do
    thandled[i] = true
    local stype = type( v )
    -- only handle value
    if stype == "table" then
       if not lookup[v] then
          table.insert( tables, v )
          lookup[v] = #tables
       end
       file:write( charS.."{"..lookup[v].."},"..charE )
    elseif stype == "string" then
       file:write(  charS..exportstring( v )..","..charE )
    elseif stype == "number" then
       file:write(  charS..tostring( v )..","..charE )
    end
 end

 for i,v in pairs( t ) do
    -- escape handled values
    if (not thandled[i]) then
    
       local str = ""
       local stype = type( i )
       -- handle index
       if stype == "table" then
          if not lookup[i] then
             table.insert( tables,i )
             lookup[i] = #tables
          end
          str = charS.."[{"..lookup[i].."}]="
       elseif stype == "string" then
          str = charS.."["..exportstring( i ).."]="
       elseif stype == "number" then
          str = charS.."["..tostring( i ).."]="
       end
    
       if str ~= "" then
          stype = type( v )
          -- handle value
          if stype == "table" then
             if not lookup[v] then
                table.insert( tables,v )
                lookup[v] = #tables
             end
             file:write( str.."{"..lookup[v].."},"..charE )
          elseif stype == "string" then
             file:write( str..exportstring( v )..","..charE )
          elseif stype == "number" then
             file:write( str..tostring( v )..","..charE )
          end
       end
    end
 end
 file:write( "},"..charE )
end
file:write( "}" )
file:close()
end

--// The Load Function
function functions.table_load( sfile )
local ftables,err = loadfile( sfile )
if err then return _,err end
local tables = ftables()
for idx = 1,#tables do
 local tolinki = {}
 for i,v in pairs( tables[idx] ) do
    if type( v ) == "table" then
       tables[idx][i] = tables[v[1]]
    end
    if type( i ) == "table" and tables[i[1]] then
       table.insert( tolinki,{ i,tables[i[1]] } )
    end
 end
 -- link indices
 for _,v in ipairs( tolinki ) do
    tables[idx][v[2]],tables[idx][v[1]] =  tables[idx][v[1]],nil
 end
end
return tables[1]
end