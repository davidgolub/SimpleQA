require('..')
local vocab_path = 'data/Translation/train/input_vocab.txt'
local hash_vocab = dmn.HashVocab(vocab_path, true)
local input_layer = dmn.HashLayer{
	emb_dim = 300,
	dropout_prob = 0.5,
	gpu_mode = false,
	dropout = false,
	vocab = hash_vocab
}
local inputs = {"La", "Foo", "bar"}
local latent_semantic = input_layer:forward(inputs)
print(latent_semantic)
local errs = input_layer:backward(inputs, latent_semantic)
print(errs)

