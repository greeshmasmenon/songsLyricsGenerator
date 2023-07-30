# test.py
import sys
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
# Verify it works
from slack_sdk import WebClient
client = WebClient()
api_response = client.api_test()

import logging
logging.basicConfig(level=logging.DEBUG)

import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

slack_token = "xoxb-5645024592213-5641400352182-tC4HW8ez40tIdGKOrlx3w2un"
#slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

try:
    response = client.chat_postMessage(
        channel="slg",
        text="Hello from your app! It is indeed working ! :tada:"
    )
except SlackApiError as e:
    # You will get a SlackApiError if "ok" is False
    assert e.response["error"]    # str like 'invalid_auth', 'channel_not_found'




from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
slack_token = "xoxb-5645024592213-5641400352182-tC4HW8ez40tIdGKOrlx3w2un"
client = WebClient(token=slack_token)

try:
    filepath="/Users/macbook/PycharmProjects/songsLyricsGenerator/src/handlers/slack_handler.py"
    response = client.files_upload(channels='#slg', file=filepath)
    assert response["file"]  # the uploaded file
except SlackApiError as e:
    # You will get a SlackApiError if "ok" is False
    assert e.response["ok"] is False
    assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
    print(f"Got an error: {e.response['error']}")