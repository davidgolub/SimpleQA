require('..')
tmp = dmn.ImageEmbedModule{
  num_classes = 1000,
  network_type = "resnet_152",
  shortcut_type = "C",
  gpu_mode = false,
  classify = false,
  load_weights = false
}

inputs = image.lena()
results = tmp:forward(inputs)

print(results)