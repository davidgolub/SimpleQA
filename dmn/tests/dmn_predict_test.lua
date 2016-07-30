require('..')
local model_save_path = "trained_models/DMN_Network_40.th"
local model = dmn.DMN_Network.load(model_save_path)
local prediction = model:predict("In French?", "The answer is far from obvious", 1)
print(prediction)