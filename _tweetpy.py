import os
import tweepy as tw
import pandas as pd
import datetime
from datetime import date
import numpy as np
import time
import auth.twitter_cred as tw_cred
import files_setup



consumer_key= tw_cred.CONSUMER_KEY
consumer_secret= tw_cred.CONSUMER_SECRET
access_token= tw_cred.ACCESS_TOKEN
access_token_secret= tw_cred.ACCESS_TOKEN_SECRET

auth = tw.AppAuthHandler(consumer_key, consumer_secret)
# auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = "#ยกเลิก112"
date_since = "2020-06-06"
num_items = 50000
# https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
lang_code = 'th'

start_time = time.time()
local_start_time = time.ctime(start_time)
print("Start Local time:", local_start_time)

# Collect tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              lang=lang_code,
              since=date_since).items(num_items)

tweets_locs = [[tweet.id, tweet.user.screen_name, tweet.user.id_str, tweet.user.location, tweet.text, tweet.created_at, tweet.user.followers_count, tweet.user.friends_count, tweet.user.favourites_count, tweet.retweet_count, tweet.favorite_count] for tweet in tweets]

tweet_text = pd.DataFrame(data=tweets_locs, columns=["tweet_id", "user", "user_id", "location", "text", "create_at", "user_follower_count", "user_friend_count", "user_favorite_count", "retweet_count", "favorite_count"])

today = date.today()
d4 = today.strftime("%b-%d-%Y")

# number_of_chunks = 1000
# for id, df_i in  enumerate(np.array_split(tweet_text, number_of_chunks)):
#     # the `id` inside {} may be omitted,
#     # I also inserted the missing closing parenthesis
#     df_i.to_csv("output/tweetsT"+d4+"N"+str(num_items)+"I"+str(id)+".csv", encoding='utf_8_sig')

dir_loc = 'output'
files_setup.setup_dir(dir_loc)
tweet_text.to_csv(dir_loc+"/tweetsT"+d4+"N"+str(num_items)+".csv", encoding='utf_8_sig')

end_time = time.time()
local_end_time = time.ctime(end_time)
print("End Local time:", local_end_time)
print("--- %s seconds ---" % (end_time - start_time))
print(str(datetime.timedelta(seconds=end_time - start_time)))