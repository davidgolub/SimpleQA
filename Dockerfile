FROM ubuntu:14.04

# Get git and curl
RUN apt-get update
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install wget -y
RUN apt-get install python-dev -y
RUN apt-get install python-pip -y

# Get python and dependencies
RUN apt-get -y install python-pip

# Need latest version of numpy, otherwise image loading fails
RUN pip install --upgrade numpy

# Logging stuff
RUN pip install cassandra-driver
RUN pip install cqlengine

# For unicode decoding
RUN pip install unidecode

# Searching stuff
RUN pip install elasticsearch

# For processing freebase data
RUN apt-get install -y pv
RUN apt-get install -y libpq-dev
# Download S3/EC2 stuff
RUN pip install tinys3
RUN pip install boto
RUN pip install filechunkio

# Run Torch7 installation scripts \

RUN apt-get install -y software-properties-common
RUN curl -sk https://raw.githubusercontent.com/davidgolub/WorkSamples/master/torch_install.sh install-deps | bash 
RUN git clone https://github.com/torch/distro.git ~/torch --recursive 
RUN cd ~/torch; ./install.sh 

# Export paths 
ENV PATH=/root/torch/install/bin:$PATH \ 
LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \ DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH 
# Set ~/torch as working directory WORKDIR /root/torch 

# Install luarocks and openssl
RUN apt-get install -y luarocks
RUN apt-get install -y openssl

# Install mysql
# RUN apt-get install -y mysql

# Install app dependencies
# RUN luarocks install luasql-mysql
RUN luarocks install mediator_lua
RUN luarocks install say
RUN luarocks install luassert
RUN luarocks install penlight
RUN luarocks install lua-term
RUN luarocks install dkjson
RUN luarocks install lua_cliargs
RUN luarocks install luasocket
RUN luarocks install copas
RUN luarocks install luasec OPENSSL_LIBDIR=/usr/lib/`gcc -print-multiarch`
RUN luarocks install busted
# RUN luarocks install fblualib
RUN luarocks install luacrypto
RUN luarocks install xxhash
RUN luarocks install lua-resty-hmac
RUN luarocks install utf8

# Async http requests and loading graphic stuff
RUN luarocks install async
RUN luarocks install nnx
RUN luarocks install torchffi
RUN luarocks install env
RUN luarocks install graphicsmagick
RUN apt-get install -y libgraphicsmagick-dev

# Install app dependencies
RUN luarocks install wsapi-xavante
RUN luarocks install xavante
RUN luarocks install cgilua
RUN luarocks install sailor
RUN luarocks install nn
RUN luarocks install nngraph
RUN luarocks install optim
RUN luarocks install json
RUN luarocks install inspect
RUN luarocks install httpclient
RUN luarocks install penlight
RUN luarocks install image
RUN luarocks install sys

# To load caffe modules from lua
RUN apt-get install -y libprotobuf-dev protobuf-compiler 

# Install load caffe module
# RUN luarocks install loadcaffe

# Install cuda/cudnn/etc
# RUN luarocks install cutorch
# RUN luarocks install cunn
# RUN luarocks install cudnn
# RUN luarocks install ccn2
# RUN luarocks install inn

# Copy apis to production 
COPY . /src

# Bundle app source

EXPOSE  8080
WORKDIR /src
CMD ["/bin/bash", "start_server.sh"]