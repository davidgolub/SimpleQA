Installation
------------

```
git clone git@github.com:theodo/flask-boilerplate.git && cd flask-boilerplate
docker-compose run --rm server pip install -r requirements.txt --user --upgrade
docker-compose up -d server
```

Accessing containers
--------------------

Require Docker >= 1.3

```shell
# use 'docker ps' to see the list of your containers
docker exec -it flaskboilerplate_db_1 psql -Upostgres
docker exec -it flaskboilerplate_server_1 bash
```

Migration process
-----------------

```shell
# Prior to the first migration
docker-compose run --rm server python src/manage.py db init

# Create a new version of the database
docker-compose run --rm server python src/manage.py db migrate
# check file + remove comment + improve file if needed
sudo vim migration/versions/<migration_id>.py

# Upgrade your database to the last version
docker-compose run --rm server python src/manage.py db upgrade
```

Run tests
---------

```shell
docker-compose run --rm server python -m unittest
```

Commands
--------

```shell
# Screenshot of python vendors
docker-compose run --rm server pip freeze > requirements.txt

# Run a command in the server container:
docker-compose run --rm server <command>
```
