local Squeeze, parent = torch.class('dmn.Squeeze', 'nn.Module')

function Squeeze:__init()
   parent.__init(self)
   self.gradInput = {}
end


function Squeeze:updateOutput(input)
  self.size = input:size()
  self.output = input:squeeze()
  return self.output
end

function Squeeze:cuda()
	dmn.logger:print("Called cuda on squeeze unit")
end

function Squeeze:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:view(self.size)
  return self.gradInput 
end