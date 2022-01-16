# getTitanicData.py
from dotenv import load_dotenv, find_dotenv

#find .env automatically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

import os
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
print(KAGGLE_USERNAME)

import requests
from requests import session
from dotenv import load_dotenv, find_dotenv

# payload for post
payload = {
    'action': 'login',
    'username': os.environ.get("KAGGLE_USERNAME"),
    'password': os.environ.get("KAGGLE_PASSWORD")
}

# kaggle competitions download -c titanic
url = 'https://www.kaggle.com/c/titnic/download/train.csv'

#setup session
with session() as c:
    # post request
    c.post('https://www.kagle.com/account/login', data=payload)
    # get response
    response = c.get(url)
    # print response text
    print(response.text)