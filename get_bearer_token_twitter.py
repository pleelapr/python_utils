import base64
import requests
import urllib.parse
import auth.twitter_cred as tw_cred

OAUTH2_TOKEN = 'https://api.twitter.com/oauth2/token'

CONSUMER_KEY= tw_cred.CONSUMER_KEY
CONSUMER_SECRET= tw_cred.CONSUMER_SECRET

def get_bearer_token(consumer_key, consumer_secret):
    # enconde consumer key
    consumer_key = urllib.parse.quote(consumer_key)
    # encode consumer secret
    consumer_secret = urllib.parse.quote(consumer_secret)
    # create bearer token
    bearer_token = consumer_key + ':' + consumer_secret
    # base64 encode the token
    base64_encoded_bearer_token = base64.b64encode(bearer_token.encode('utf-8'))
    # set headers
    headers = {
        "Authorization": "Basic " + base64_encoded_bearer_token.decode('utf-8') + "",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Content-Length": "29"}

    response = requests.post(OAUTH2_TOKEN, headers=headers, data={'grant_type': 'client_credentials'})
    to_json = response.json()
    print("token_type = %s" % (to_json['token_type']))
    print("access_token  = %s" % (to_json['access_token']))

if __name__ == "__main__":
    get_bearer_token(CONSUMER_KEY, CONSUMER_SECRET)