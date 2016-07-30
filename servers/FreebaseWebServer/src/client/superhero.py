"""
This is a client of another fake API.

It is used in this boilerplate to show how to mock an external API and how to
chaine errors between APIs.
"""
import requests

import config

def is_superhero(email):
    response = requests.get(config.SUPERHERO_API_URL + '/superhero/%s' % email)

    if response.status_code == 404:
        return False
    elif response.status_code == 200:
        return True
    else:
        raise Exception('An error occured in the superhero API')
