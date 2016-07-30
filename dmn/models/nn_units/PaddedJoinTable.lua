local JoinTable, parent = torch.class('dmn.PaddedJoinTable', 'nn.Module')

function JoinTable:__init()
   parent.__init(self)
   self.size = torch.LongStorage()
   self.gradInput = {}
   self.nInputDims = nInputDims
end

function JoinTable:updateOutput(input)
   local num_input = #input
   local input_size = input[1]:size(1)
   self.size = torch.LongStorage{num_input, input_size}

   local output = self.output:resize(self.size)
   for i = 1, #input do 
    output[i]:copy(input[i])
   end
   return output
end

function JoinTable:updateGradInput(input, gradOutput)
   local dimension = self.dimension
   if self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
       dimension = dimension + 1
   end

   for i=1,#input do
      if self.gradInput[i] == nil then
         self.gradInput[i] = input[i].new()
      end
      self.gradInput[i]:resizeAs(input[i])
   end

   -- clear out invalid gradInputs
   for i=#input+1, #self.gradInput do
      self.gradInput[i] = nil
   end

   for i=1,#input do
      local currentGradInput = gradOutput[i]
      self.gradInput[i]:copy(currentGradInput)
   end
   return self.gradInput
end

function JoinTable:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end
