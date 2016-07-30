require('..')

gru_decoder_model = dmn.GRU_Decoder{ 
            gpu_mode = false,
            in_dim = 150,
            mem_dim = 100,
            dropout_prob = 0.5,
            dropout = 0.5,
            num_layers = 1 
            }

gru_encoder_model = dmn.GRU_Encoder{ 
            gpu_mode = false,
            in_dim = 150,
            mem_dim = 100,
            dropout_prob = 0.5,
            dropout = 0.5,
            num_layers = 1 
            }

gru_answer_module = dmn.AnswerModule{
  dropout = false,
  in_dropout_prob = 0.0,
  hidden_dropout_prob = 0.0,
  gpu_mode = false,
  num_classes = 1000,
  emb_dim = 100,
  input_dim = 100,
  mem_dim = 150,
  num_layers = 1
}

input = torch.IntTensor{1, 2, 3, 4, 5,}
output = torch.IntTensor{5, 3, 2, 1, 6}

input_decoder = torch.rand(5, 100)
memory = torch.rand(150)               
--decoder_results = gru_decoder_model:forward(input_decoder, memory, false)

err = gru_answer_module:forward(memory, input, output)
bprop = gru_answer_module:backward(memory, input, output)
print(err)
print(bprop)
