curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
luarocks install json
luarocks install nn
luarocks install nngraph
luarocks install json
luarocks install luasocket
luarocks install lua-cjson
luarocks install luasec
luarocks install utf8
luarocks install threads
luarocks install argcheck
luarocks install xavante
luarocks install wsapi-xavante
luarocks install cgilua
luarocks install sailor

pip install tinys3

# Core utils for mac
brew install coreutils findutils gnu-tar gnu-sed gawk gnutls gnu-indent gnu-getopt


git filter-branch --prune-empty -d ~/Desktop/scratch \
  --index-filter "git rm --cached -f --ignore-unmatch TorchWebServer/cpu/softmax/models/cm_captioning_5.th" \
  --tag-name-filter cat -- --all

# Processes:
14086