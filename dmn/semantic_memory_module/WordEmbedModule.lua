--[[

  A WordEmbedModule takes two things as input: 
  1) a lookup-table for word embeddings
  
  It encodes word indices to embeddings which could either be hash 
  indices or hash + LSTM indices

--]]

local WordEmbedModule = torch.class('dmn.WordEmbedModule')

function WordEmbedModule:__init(config)
 -- parameters for lstm cell
  assert(config.gpu_mode ~= nil, "Must specify gpu mode")
  assert(config.emb_dim ~= nil, "Must specify embed dimensions")
  assert(config.num_classes ~= nil, "Must specify number of classes")
  assert(config.dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.hashing ~= nil, "Must specify whether to hash or not")
  assert(config.dropout ~= nil, "Must specify whether to use dropout or not")
  
  self.config = config
  self.hashing = config.hashing
  self.gpu_mode = config.gpu_mode
  self.reverse = false;

  local embed_type = self.hashing and dmn.FastHashLayer or dmn.EmbedLayer

  self.embed_layer = embed_type{
                      gpu_mode = config.gpu_mode,
                      emb_dim = config.emb_dim,
                      num_classes = config.num_classes,
                      dropout_prob = config.dropout_prob,
                      dropout = config.dropout
                    }

  self.modules = nn.Parallel()
  add_modules(self.modules, self.embed_layer:getModules())

  if self.gpu_mode then
    dmn.logger:print("Setting word embed module to gpu mode")
    self:set_gpu_mode()
  end

  self.tot_modules = {}
  insert_modules_to_table(self.tot_modules, self.embed_layer:getModules())

  --self.params, self.grad_params = self.modules:getParameters()

  dmn.logger:print("Modules we're optimizing for word embed module")
  dmn.logger:print(self.modules)
end

function WordEmbedModule:share(other, ...)
  assert(other ~= nil, "Must specify other embed layer to share")
  dmn.logger:print("Sharing word embed module")
  self.embed_layer:share_params(other.embed_layer, ...)
end

-- Enable Dropouts
function WordEmbedModule:enable_dropouts()
  self.embed_layer:enable_dropouts()
end

-- Disable Dropouts
function WordEmbedModule:disable_dropouts()
  self.embed_layer:disable_dropouts()
end

-- Resets depth to 1
function WordEmbedModule:reset_depth()
end


function WordEmbedModule:zeroGradParameters()
  self.grad_params:zero()
  self.embed_layer:zeroGradParameters()
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns lstm output, class predictions, and error if train, else not error 
function WordEmbedModule:forward(inputs)
    assert(inputs ~= nil, "Must specify inputs to forward for word embed module")
    self.word_embeds = self.embed_layer:forward(inputs)
    return self.word_embeds
end


-- Backpropagate: forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- hidden_inputs: {hidden_dim, hidden_tim} tensors
-- reverse: True if reverse input, false otherwise
-- errors: T x input_size error
-- class_predictions: T x 1 tensor of predictions
-- labels: actual labels
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function WordEmbedModule:backward(inputs, errors)
  assert(inputs ~= nil, "Must put in gru regular inputs")
  assert(errors ~= nil, "must put in lstm outputs")
  local emb_errors = self.embed_layer:backward(inputs, errors)
  return emb_errors
end

function WordEmbedModule:grad_check()
  self.params, self.grad_params = self.modules:getParameters()
  local input_indices = torch.IntTensor{1, 2, 3}
  local criterion = nn.MSECriterion()
  local desired_state = torch.rand(3, self.lstm_layer.mem_dim)

  local currIndex = 0
  local feval = function(x)
      self.grad_params:zero()
      local lstm_output = self:forward(input_indices)
      local loss = criterion:forward(lstm_output, desired_state)
      local errors = criterion:backward(lstm_output, desired_state)
      self:backward(input_indices, errors)
      currIndex = currIndex + 1
      print(currIndex, " of ", self.params:size(1))    
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    print("Gradient error for word embed module network is")
    print(diff)
    assert(diff < 1e-5, "Gradient is greater than tolerance")
end

-- Sets all networks to gpu mode
function WordEmbedModule:set_gpu_mode()
  dmn.logger:print("Setting word embed module to gpu mode")
  self.gpu_mode = true
  self.config.gpu_mode = true
  self.embed_layer.gpu_mode = true
  self.embed_layer:set_gpu_mode()
end

-- Sets all networks to cpu mode
function WordEmbedModule:set_cpu_mode()
  dmn.logger:print("Setting word embed module to cpu mode")
  self.gpu_mode = false
  self.config.gpu_mode = false
  self.embed_layer.gpu_mode = false
  self.embed_layer:set_cpu_mode()
end

function WordEmbedModule:getModules() 
  return self.tot_modules
end

function WordEmbedModule:getParameters()
  return self.params, self.grad_params
end

function WordEmbedModule:getWeights()
  return self.params
end

-- Resets depths for lstm
function WordEmbedModule:forget()
end

function WordEmbedModule:print_config()
  local num_params = 0 --self.params:size(1)
  printf('%-25s = %d\n', 'num params for word embed module', num_params)
  printf('%-25s = %s\n', 'gpu mode', self.config.gpu_mode)
  printf('%-25s = %s\n', 'use dropout', self.config.dropout)
  printf('%-25s = %d\n', 'embed dimension', self.config.emb_dim)
  printf('%-25s = %d\n', 'number of classes', self.config.num_classes)
  printf('%-25s = %f\n', 'input dropout probability', self.config.dropout_prob)
end

