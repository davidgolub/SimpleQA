 require('..')

 -- creates soft attention
 local context_size = 2
 local rnn_size = 2
 local num_layers = 1
 local context_size = 2
 local input = nn.Identity()()

  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()
  
  local context = nn.Identity()()
  local first_h_layer = (num_layers == 1) and htable_p or nn.SelectTable(1)(htable_p)
  -- Attention is softmax(e_ij) where e_ij = va^T * tanh(Wa*S_i-1 + U_a * h_j)
  local perceptroned_context = nn.Linear(context_size, rnn_size)(context)
  local added_context = nn.Tanh()(dmn.CRowAddTable(){perceptroned_context, nn.Linear(rnn_size, rnn_size)(first_h_layer)})
  local soft_attention = nn.SoftMax()(dmn.Squeeze()(nn.Linear(rnn_size, 1)(added_context)))
  local replicated_attention = nn.Replicate(context_size, 2)(soft_attention)
  local summed_context = nn.Sum()(nn.CMulTable(){replicated_attention, context})

  net = nn.gModule({context, htable_p}, {soft_attention, replicated_attention, summed_context})

  local res = net:forward({torch.rand(3, 2), torch.rand(2)})
  print(res[1])
  print(res[2])
  print(res[3])