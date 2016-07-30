require('..')
require('cunn')
require('cutorch')

model = dmn.Context_Captioner_Network.load('trained_models/Context_Captioner_Network_1.th')
model:set_gpu_mode()
img = image.lena()
context = "The cat ran over the board"
beam_size = 1

model:save("test.th", 1)
new_model = dmn.Context_Captioner_Network.load("test.th")
new_model:set_gpu_mode()
results = model:predict(img, context, beam_size)

new_results = new_model:predict(img, context, beam_size)

img_embed_diff = new_model.image_embed_layer:forward(img)
img_embed_prev = model.image_embed_layer:forward(img)

context = torch.CudaTensor{1, 2, 3, 4, 5}

loss1, class_predictions1 = new_model.answer_layer:forward(img_embed_diff, context, context, context)
loss, class_predictions = model.answer_layer:forward(img_embed_prev, context, context, context)

cur_params = model.params
new_params = new_model.params

local diff = cur_params-new_params
print("Params diff")
print(diff:sum())
print(loss)
print(loss1)
print(class_predictions1)
print(class_predictions)

diff = img_embed_diff - img_embed_prev
print(diff:sum())
print(results)