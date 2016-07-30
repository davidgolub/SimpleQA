require('.')
hashed_items = {}
items = "#" .. "foo-bar" .. "#"
for i = 1, #items - 2 do
	table.insert(hashed_items, items:sub(i, i + 2))
end

print (hashed_items)

local dir = 'data/Translation/train/'
local input_vocab = dmn.HashVocab(dir .. 'input_vocab.txt', true)
local items = input_vocab:index("cats")

print(items)
print("Done")