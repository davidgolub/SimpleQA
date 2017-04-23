#!/bin/bash
docker pull davidgolub/simpleqa .
docker run -it simpleqa /bin/bash

# Then inside the container start the freebase server
cd /src/servers/FreebaseWebServer
pip install -r requirements.txt
cd src
python server.py & disown

# Run the evaluation script
cd /src/datasets
th scripts/evaluate_sq_fb_dataset_v2.lua