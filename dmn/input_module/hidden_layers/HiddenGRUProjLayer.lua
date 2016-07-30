--[[

  Hidden Project Layer: Projects image input into projection dimension twice. For feeding in
  image input into lstm
--]]

local HiddenProjLayer, parent = torch.class('dmn.HiddenGRUProjLayer', 'dmn.HiddenLayer')

function HiddenProjLayer:__init(config)
   assert(config.mem_dim ~= nil, "Must specify memory dimensions")
   assert(config.input_dim ~= nil, "Must specify input dimensions")
   assert(config.dropout_prob ~= nil, "Must specify dropout probability")
   assert(config.num_layers ~= nil, "Must specify number of layers")
   assert(config.dropout ~= nil, "Must specify dropout")

   self.gpu_mode = config.gpu_mode
   self.input_dim = config.input_dim
   self.proj_dim = config.mem_dim
   self.dropout_prob = config.dropout_prob
   self.dropout = config.dropout
   self.num_layers = config.num_layers
   
   local modules = nn.Parallel()
   -- image feature embedding
   if self.num_layers == 1 then 
    local hidden_emb = self:new_hidden_module()
    self.hidden_emb = hidden_emb
    modules:add(self.hidden_emb)

   else
    self.hidden_emb = {}
    for i = 1, self.num_layers do
      local hidden_emb = self:new_hidden_module()
      table.insert(self.hidden_emb, hidden_emb)

      modules:add(self.hidden_emb[i])
    end
   end
   

  if gpu_mode then
    self:set_gpu_mode()
  end
end

function HiddenProjLayer:new_hidden_module() 
  local hidden_emb = nn.Sequential() 
        :add(nn.Linear(self.input_dim, self.proj_dim))

  if self.dropout then
      hidden_emb:add(nn.Dropout(self.dropout_prob, false))
  end
  return hidden_emb
end

-- Returns all of the weights of this module
function HiddenProjLayer:getWeights()
  return self.params
end

function HiddenProjLayer:getModules() 
  if self.num_layers == 1 then 
    return {self.hidden_emb}
  else 
    local modules = {}
    for i = 1, self.num_layers do
      table.insert(modules, self.hidden_emb[i])
    end
    return modules
  end
end

-- Sets gpu mode
function HiddenProjLayer:set_gpu_mode()
  self.gpu_mode = true
  if self.num_layers == 1 then 
     self.hidden_emb:cuda()
  else 
    for i = 1, self.num_layers do
       self.hidden_emb[i]:cuda()
    end
  end
end

function HiddenProjLayer:set_cpu_mode()
  self.gpu_mode = false
  if self.num_layers == 1 then 
     self.hidden_emb:double()
  else 
    for i = 1, self.num_layers do
       self.hidden_emb[i]:double()
    end
  end
end

-- Enable Dropouts
function HiddenProjLayer:enable_dropouts()
  if self.num_layers == 1 then 
    enable_sequential_dropouts(self.hidden_emb)
  else 
    for i = 1, self.num_layers do
      enable_sequential_dropouts(self.hidden_emb[i])
    end
  end
end

-- Disable Dropouts
function HiddenProjLayer:disable_dropouts()
  if self.num_layers == 1 then 
    disable_sequential_dropouts(self.hidden_emb)
  else 
    for i = 1, self.num_layers do
      disable_sequential_dropouts(self.hidden_emb[i])
    end
  end
end

-- Does a single forward step of concat layer, concatenating
-- 
function HiddenProjLayer:forward(inputs)
   assert(inputs ~= nil, "Hidden inputs are null")
   local ndim = inputs:dim()

   assert(inputs:size(ndim) == self.input_dim, 
    "Dimension mismatch on hidden inputs " .. " expected " .. self.input_dim)
   parent:forward(inputs, self.gpu_mode)

   if self.num_layers == 1 then
     self.hidden_image_proj = self.hidden_emb:forward(inputs)
     return self.hidden_image_proj
   else
     local hidden_vals = {}

     for i = 1, self.num_layers do
      local hidden_image_proj = self.hidden_emb[i]:forward(inputs)
      table.insert(hidden_vals, hidden_image_proj)
     end

     return hidden_vals
   end
   
end

-- Does a single backward step of project layer
-- inputs: input into hidden projection error
-- cell_errors: error of all hidden, cell units of lstm with respect to input
function HiddenProjLayer:backward(inputs, cell_errors)
   assert(inputs ~= nil)
   assert(inputs:size(inputs:dim()) == self.input_dim, 
    "Dimension mismatch on hidden inputs " .. " expected " 
    .. self.input_dim)
   assert(cell_errors ~= nil)
   
   local input_errors = torch.zeros(inputs:size())
   if self.num_layers == 1 then
     -- get the image and word projection errors
     local hidden_emb_errors = self.hidden_emb:backward(inputs, cell_errors)
     input_errors = hidden_emb_errors
   else
     for i = 1, self.num_layers do
        -- get the image and word projection errors
       local hidden_emb_errors = cell_errors[i]
       
       -- feed them backward
       local hidden_input_errors = self.hidden_emb[i]:backward(inputs, hidden_emb_errors)
       input_errors = hidden_input_errors + input_errors
     end
   end
   return input_errors
end

-- Returns size of outputs of this combine module
function HiddenProjLayer:getOutputSize()
  return self.mem_dim
end

function HiddenProjLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function HiddenProjLayer:zeroGradParameters() 
  if self.num_layers == 1 then
    self.cell_emb:zeroGradParameters()
    self.hidden_emb:zeroGradParameters()
  else
    for i = 1, self.num_layers do
      self.cell_emb[i]:zeroGradParameters()
      self.hidden_emb[i]:zeroGradParameters()
    end
  end
end

function HiddenProjLayer:normalizeGrads(batch_size)
  assert(batch_size ~= nil)
  if self.num_layers == 1 then
    self.cell_emb.gradWeight:div(batch_size)
    self.hidden_emb.gradWeight:div(batch_size)
  else
    for i = 1, self.num_layers do
      self.cell_emb[i].gradWeight:div(batch_size)
      self.hidden_emb[i].gradWeight:div(batch_size)
    end
  end
end

