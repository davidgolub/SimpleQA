require('..')
dummy_path = 'data/QA/vocab.txt'
vocab_path = 'data/QA/inputs.txt'

dataset = dmn.read_dataset('data/QA')
print(dataset)