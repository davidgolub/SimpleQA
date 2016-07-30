local functions = torch.class('dmn.math_functions')

function functions.copy_lstm_units(hidden_inputs, num_layers)
  assert(hidden_inputs ~= nil, "Must specify num hidden inputs")
  assert(num_layers ~= nil, "Must specify number of layers")
  local new_inputs
  if num_layers > 1 then
    -- Array of cell states and hidden states
    new_inputs = {{}, {}}
    for i = 1, #hidden_inputs[1] do

      new_inputs[1][i] = hidden_inputs[1][i]:clone():zero()
      new_inputs[2][i] = hidden_inputs[2][i]:clone():zero()
    end
  else
    new_inputs = {hidden_inputs[1]:clone():zero(), hidden_inputs[2]:clone():zero()}
  end
  return new_inputs
end

function functions.copy_gru_units(hidden_inputs, num_layers)
  assert(hidden_inputs ~= nil, "Must specify num hidden inputs")
  assert(num_layers ~= nil, "Must specify number of layers")
  local new_inputs
  if num_layers > 1 then
    new_inputs = {}
    for i = 1, #hidden_inputs do
      new_inputs[i] = hidden_inputs[i]:clone():zero()
    end
  else 
    new_inputs = hidden_inputs:clone():zero()
  end
  return new_inputs
end
-- Gets learning rate with epoch and current learning rate
function functions.get_learning_rate(learning_rate, epoch)
   assert(learning_rate ~= nil, "Must specify learning rate")
   assert(epoch ~= nil, "Must specify epoch")

   local decay = math.floor((epoch - 1) / 30)
   return learning_rate * math.pow(0.1, decay)
end

-- Argmax: hacky way to ignore end token to reduce silly sentences
function functions.argmax(v)
  assert(v ~= nil, "Must specify v to sort")
  local vals, indices = torch.max(v, 1)
  return indices[1]
end

function functions.random(k, a, b)
  local random_nums = {}
  for i = 1, k do
    local cur_num = torch.random(a, b)
    table.insert(random_nums, cur_num)
  end
  return random_nums
end

function functions.mask_gradients(inputs, batched_grads, masks)
  assert(inputs ~= nil, "Must specify inputs that correspond gradients to mask")
  assert(batched_grads ~= nil, "Must specify batched gradients to mask")

  local reshaped_grads = functions.reshape_grads(inputs, batched_grads)
  local masked_grads = functions.mask_grads(inputs, reshaped_grads, masks)
  local reshaped_masked_grads = functions.unshape_grads(masked_grads)

  return reshaped_masked_grads
  --return batched_grads
end

function functions.unshape_grads(inputs)
  assert(inputs ~= nil, "Must specify input to forward")
  local output
  if inputs:dim() > 2 then 
    local inputSize = inputs:size()
    local nInputs = inputSize:size(1)

    local newDim = inputs:size()
    newDim[nInputs] = -1
    local dimReshape = 1
    for i = 1, nInputs - 1 do
      dimReshape = dimReshape * inputSize[i]
    end

    output = inputs:view(dimReshape, -1)
  else 
    output = inputs
  end
  return output
end

function functions.mask_grads(input, gradOutput, masks)
  assert(input ~= nil, "Must specify which input to use")
  assert(gradOutput ~= nil, "Must specify gradOutput to use")
  local gradInput
  if input:dim() > 2 then 
    assert(masks ~= nil, "Must specify masks to use for batch input")
      gradInput = gradOutput
    for i = 1, masks:size(1) do
      local cur_mask = masks[i] 
      local start_index = cur_mask[1]
      local end_index = cur_mask[2]

      if start_index <= end_index then 
        gradInput[{{start_index, end_index},{i,i}}]:zero()
      end
    end 
  else 
    gradInput = gradOutput
  end
    return gradInput 
end

function functions.reshape_grads(inputs, gradOutput)
  assert(inputs ~= nil, "Must specify input to backward")
  assert(gradOutput ~= nil, "Must specify gradOutput to use")
  local gradInput 
  if inputs:dim() > 2 then 
    local inputSize = inputs:size()
    local nInputs = inputSize:size(1)

    local newDim = inputs:size()
    newDim[nInputs] = -1

    -- softmax criterion has gradient = to mean of sums.
    -- we need to amplify it back up again to match sequential case.
    gradInput = gradOutput:view(newDim) * inputs:size(2)
  else 
    gradInput = gradOutput
  end
  return gradInput 
end

function functions.check_layers()
  if functions.reshape_layer == nil then 
    functions.reshape_layer = dmn.BatchReshape()
  end 

  if functions.mask_layer == nil then 
    functions.mask_layer = dmn.BatchMask()
  end
end


-- Returns equality if two tensors are equal, false otherwise
function functions.tensors_equals(tensor1, tensor2)
  if torch.typename(tensor1) ~= torch.typename(tensor2) then
    return false
  end

  if tensor1:dim() ~= tensor2:dim() then
    return false
  end

  for i = 1, tensor1:dim() do 
    if tensor1:size(i) ~= tensor2:size(i) then
      return false
    end
  end

  local diff = tensor1 - tensor2
  return diff:sum() == 0
end

function functions.equals(obj1, obj2)
  -- check if it's a table
  if torch.type(obj1) ~= torch.type(obj2) then
    return false
  end

  -- if it's a table, iterate through the keys
  if torch.type(obj1) == torch.type({}) then
    if #obj1 ~= #obj2 then
      return false
    end
    -- move optim state to double as well
    for k,v in pairs(obj1) do 
      if obj2[k] == nil then 
        return false
      end
      if torch.typename(v) ~= nil then 
        local tensors_equal = functions.tensors_equals(obj1[k], obj2[k])
        if not tensors_equal then return false end
      end
    end
    return true
  elseif torch.typename(obj1) ~= nil then 
    if torch.typename(obj1) ~= torch.typename(obj2) then
      return false
    else
      return functions.tensors_equals(obj1, obj2)
    end
  else
    return obj1 == obj2
  end
end

-- Returns a DoubleTensor, where ith entry is likelihood of sampling ith element of values
function functions.sampling_probability(values)
  assert(values ~= nil, "Must specify dictionary to sample from")

  local num_items = #values
  local sampling_tensor = torch.DoubleTensor(num_items)
  local dict_to_count = {}
  for i = 1, num_items do
    local cur_val = values[i] 
    if dict_to_count[cur_val] == nil then 
      dict_to_count[cur_val] = 1
    else 
      dict_to_count[cur_val] = dict_to_count[cur_val] + 1
    end
  end

  for i = 1, num_items do 
    local cur_val = values[i]
    sampling_tensor[i] = dict_to_count[cur_val]
  end
  sampling_tensor = sampling_tensor / num_items
  return sampling_tensor
end

-- TopkArgmax returns top k indices, values from list
function functions.topkargmax(list, k)
  assert(list ~= nil, "Must specify list to sort")
  assert(k ~= nil, "Must specify number of top choices to pick")

  local cloned_list = list:clone()
  max_indices = {}
  max_probabilities = {}
  for i = 1, k do
    local vals, indices = torch.max(cloned_list, 1)
    local best_index = indices[1]
    local best_probability = list[best_index]

    cloned_list[best_index] = -1000
    table.insert(max_indices, best_index)
    table.insert(max_probabilities, best_probability)
  end
  return max_indices, max_probabilities
end

-- Creates a label which is a probability distribution of labels interpolated with "gold label"
-- i.e. prob = actual_label * ratio + (1-ratio) * probability_dist[k] / sum(probability_dist[k])
-- actual_label: "gold label" of class, i.e. 1
-- probability_dist: Probability distribution to interpolate with 
-- num_classes: Number of classes to take from distribution (if num_classes == -1 then take all)
-- ratio: Ratio of interpolation between actual_label and probabilities
function functions.probability_interpolation(actual_label, probability_dist, num_classes, ratio)
  assert(actual_label ~= nil, "Must specify actual label")
  assert(probability_dist ~= nil, "Must specify probability_dist")
  assert(num_classes ~= nil, "Must specify number number of classes to do transfer learning from")
  assert(ratio ~= nil, "Must specify ratio to do transfer learning from")

  local interpolated_result 
  local class_index = actual_label
  if num_classes == -1 then 
    interpolated_result = probability_dist * (1 - ratio)
    interpolated_result[class_index] = interpolated_result[class_index] + ratio
  else
    -- sample top k
    max_indices, max_probabilities = functions.topkargmax(probability_dist, num_classes)
    interpolated_result = probability_dist:clone():zero()

    -- compute sum of max probabilities
    local cum_sum = 0
    for i = 1, #max_indices do 
      local cur_index = max_indices[i]
      local cur_prob = max_probabilities[i]

      interpolated_result[cur_index] = cur_prob
      cum_sum = cum_sum + cur_prob
    end
    
    -- Do we have to watch out for double underfloat?
    interpolated_result = interpolated_result * (1-ratio) / cum_sum

    interpolated_result[class_index] = interpolated_result[class_index] + ratio
  end

  return interpolated_result
end