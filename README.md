This contains the code corresponding to the publication Character-Level Question-Answering with Attention found at:
https://arxiv.org/abs/1604.00727 

To run the evaluation script first build the container:
# Building the container
docker build -t simpleqa .

docker run -it simpleqa /bin/bash

# Then inside the container start the freebase server
cd /src/servers/FreebaseWebServer

pip install -r requirements.txt

cd src

python server.py & disown

# Run the evaluation script
cd /src/datasets

th scripts/evaluate_sq_fb_dataset_v2.lua

We hope this code helps researchers explore character-level encoder-decoder frameworks to semantic parsing with KBs and many other tasks in NLP! Email golubd@stanford.edu if you have any questions.

