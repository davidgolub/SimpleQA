--[[

  Hidden Project Layer: Projects image input into projection dimension twice. For feeding in
  image input into lstm
--]]

local HiddenProjLayer, parent = torch.class('dmn.HiddenProjLayer', 'dmn.HiddenLayer')

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
    local cell_emb, hidden_emb = self:new_hidden_module()
    self.cell_emb = cell_emb
    self.hidden_emb = hidden_emb

    modules:add(self.cell_emb)
    modules:add(self.hidden_emb)
    
   else
    self.cell_emb = {}
    self.hidden_emb = {}
    for i = 1, self.num_layers do
      local cell_emb, hidden_emb = self:new_hidden_module()
      table.insert(self.cell_emb, cell_emb)
      table.insert(self.hidden_emb, hidden_emb)

      modules:add(self.cell_emb[i])
      modules:add(self.hidden_emb[i])
    end
   end
   
   self.params, self.grad_params = modules:getParameters()

  if gpu_mode then
    self:set_gpu_mode()
  end
end

function HiddenProjLayer:new_hidden_module() 
  local cell_emb = nn.Sequential()
        :add(nn.Linear(self.input_dim, self.proj_dim))
        --:add(nn.Tanh())
  local hidden_emb = nn.Sequential() 
        :add(nn.Linear(self.input_dim, self.proj_dim))
        --:add(nn.Tanh())

  if self.dropout then
      cell_emb:add(nn.Dropout(self.dropout_prob, false))
      hidden_emb:add(nn.Dropout(self.dropout_prob, false))
  end

  if self.gpu_mode then 
    cell_emb:cuda()
    hidden_emb:cuda()
  end
  return cell_emb, hidden_emb
end

-- Returns all of the weights of this module
function HiddenProjLayer:getWeights()
  return self.params
end

function HiddenProjLayer:getModules() 
  if self.num_layers == 1 then 
    return {self.cell_emb, self.hidden_emb}
  else 
    local modules = {}
    for i = 1, self.num_layers do
      table.insert(modules, self.cell_emb[i])
      table.insert(modules, self.hidden_emb[i])
    end
    return modules
  end
end

-- Sets gpu mode
function HiddenProjLayer:set_gpu_mode()
  self.gpu_mode = true
  if self.num_layers == 1 then 
     self.cell_emb:cuda()
     self.hidden_emb:cuda()
  else 
    for i = 1, self.num_layers do
       self.cell_emb[i]:cuda()
       self.hidden_emb[i]:cuda()
    end
  end
end

function HiddenProjLayer:set_cpu_mode()
  self.gpu_mode = false
  if self.num_layers == 1 then 
     self.cell_emb:double()
     self.hidden_emb:double()
  else 
    for i = 1, self.num_layers do
       self.cell_emb[i]:double()
       self.hidden_emb[i]:double()
    end
  end
end

-- Enable Dropouts
function HiddenProjLayer:enable_dropouts()
  if self.num_layers == 1 then 
    enable_sequential_dropouts(self.cell_emb)
    enable_sequential_dropouts(self.hidden_emb)
  else 
    for i = 1, self.num_layers do
      enable_sequential_dropouts(self.cell_emb[i])
      enable_sequential_dropouts(self.hidden_emb[i])
    end
  end
end

-- Disable Dropouts
function HiddenProjLayer:disable_dropouts()
  if self.num_layers == 1 then 
    disable_sequential_dropouts(self.cell_emb)
    disable_sequential_dropouts(self.hidden_emb)
  else 
    for i = 1, self.num_layers do
      disable_sequential_dropouts(self.cell_emb[i])
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
    "Dimension mismatch on hidden inputs " .. inputs:size(ndim) .. " expected " .. self.input_dim)
   parent:forward(inputs, self.gpu_mode)

   if self.num_layers == 1 then
     self.cell_image_proj = self.cell_emb:forward(inputs)
     self.hidden_image_proj = self.hidden_emb:forward(inputs)
     return {self.cell_image_proj, self.hidden_image_proj}
   else
     local cell_vals = {}
     local hidden_vals = {}

     for i = 1, self.num_layers do
      local cell_image_proj = self.cell_emb[i]:forward(inputs)
      local hidden_image_proj = self.hidden_emb[i]:forward(inputs)

      table.insert(cell_vals, cell_image_proj)
      table.insert(hidden_vals, hidden_image_proj)
     end

     return {cell_vals, hidden_vals}
   end
   
end

-- Does a single backward step of project layer
-- inputs: input into hidden projection error
-- cell_errors: error of all hidden, cell units of lstm with respect to input
function HiddenProjLayer:backward(inputs, cell_errors)
   assert(inputs ~= nil, "Must specify inputs for hidden projection layer")
   assert(cell_errors ~= nil, "Must specify cell errors for hidden project layer")
   parent:backward(inputs, cell_errors, self.gpu_mode)
   
   local ndim = inputs:dim()
   local image_errors = torch.zeros(inputs:size())
   if self.num_layers == 1 then
     -- get the image and word projection errors
     local cell_emb_errors = cell_errors[1]
     local hidden_emb_errors = cell_errors[2]

     assert(cell_emb_errors:size(ndim) == self.proj_dim)
     assert(hidden_emb_errors:size(ndim) == self.proj_dim)

     -- feed them backward
     local cell_image_errors = self.cell_emb:backward(inputs, cell_emb_errors)
     local hidden_image_errors = self.hidden_emb:backward(inputs, hidden_emb_errors)
     image_errors = cell_image_errors + hidden_image_errors
   else
     for i = 1, self.num_layers do
        -- get the image and word projection errors
       local cell_emb_errors = cell_errors[1][i]
       local hidden_emb_errors = cell_errors[2][i]

       assert(cell_emb_errors:size(ndim) == self.proj_dim)
       assert(hidden_emb_errors:size(ndim) == self.proj_dim)

       -- feed them backward
       local cell_image_errors = self.cell_emb[i]:backward(inputs, cell_emb_errors)
       local hidden_image_errors = self.hidden_emb[i]:backward(inputs, hidden_emb_errors)
       image_errors = image_errors + cell_image_errors + hidden_image_errors
     end
   end
   return image_errors
end

-- Returns size of outputs of this combine module for a single output
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

