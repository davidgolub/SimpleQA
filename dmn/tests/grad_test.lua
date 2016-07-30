require('..')

document_embed_module = dmn.DocumentEmbedModule{ 
          gpu_mode = false,
          emb_dim = 5,
          num_classes = 10,
          dropout_prob = 0.5,
          mem_dim = 3,
          num_layers = 5,
          dropout = false
}

-- single layered gru answer module
single_layer_gru_answer_module = dmn.AnswerModule{
  dropout = false,
  in_dropout_prob = 0.0,
  hidden_dropout_prob = 0.0,
  gpu_mode = false,
  num_classes = 30,
  emb_dim = 5,
  input_dim = 2,
  mem_dim = 3,
  num_layers = 1
}

-- multilayered gru answer module
multi_layer_gru_answer_module = dmn.AnswerModule{
  dropout = false,
  in_dropout_prob = 0.0,
  hidden_dropout_prob = 0.0,
  gpu_mode = false,
  num_classes = 30,
  emb_dim = 5,
  input_dim = 2,
  mem_dim = 3,
  num_layers = 10
}

-- singlelayered question module
single_layer_question_module = dmn.QuestionModule{
  dropout = false,
  dropout_prob = 0.0,
  gpu_mode = false,
  num_classes = 1000,
  emb_dim = 2,
  mem_dim = 3,
  num_layers = 1
}

-- singlelayered question module
multi_layer_question_module = dmn.QuestionModule{
  dropout = false,
  dropout_prob = 0.0,
  gpu_mode = false,
  num_classes = 1000,
  emb_dim = 2,
  mem_dim = 3,
  num_layers = 10
}

document_embed_module:grad_check()
single_layer_gru_answer_module:grad_check()
single_layer_question_module:grad_check()
multi_layer_question_module:grad_check()
multi_layer_gru_answer_module:grad_check()


