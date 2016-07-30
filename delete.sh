#!/bin/bash

set -o errexit

echo "Removing exited docker containers..."
docker ps -a -f status=exited -q | xargs -r docker rm -v

echo "Removing dangling images..."
docker images --no-trunc -q -f dangling=true | xargs -r docker rmi

echo "Removing unused docker images"
images=($(docker images --digests | tail -n +2 | awk '{ img_id=$1; if($2!="<none>")img_id=img_id":"$2; if($3!="<none>") img_id=img_id"@"$3; print img_id}'))
containers=($(docker ps -a | tail -n +2 | awk '{print $2}'))

containers_reg=" ${containers[*]} "
remove=()

for item in ${images[@]}; do
  if [[ ! $containers_reg =~ " $item " ]]; then
    remove+=($item)
  fi
done

remove_images=" ${remove[*]} "

echo ${remove_images} | xargs -r docker rmi
echo "Done"

docker stop $(docker ps -a -q)
docker kill $(sudo docker ps -a -q)