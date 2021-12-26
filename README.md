# hatespeechHAN-telegram-bot

Hate speech detection telegram bot using Hierarchical Attention Networks (HAN)

<!--#![hatespeechbot](https://user-images.githubusercontent.com/53829167/95406020-5c936b00-0954-11eb-9ba7-f9110b95b2cb.png)-->
<div>
<img align="right" style="float: right;" width="200"src='https://user-images.githubusercontent.com/53829167/141665448-b7a9fe01-7637-46ed-ba05-f9558d31ad58.jpeg'>
 <img align="right" style="float: right;" width="200"src='https://user-images.githubusercontent.com/53829167/141665493-4eace6b8-c1ff-4436-9102-40f57dc41aea.jpeg'>

</div>

### Features

- Detect whether the sentence is hate speech or not
- Explain why the machine came to judge the sentence as hate speech.

We presented an example sentence and an output image together to explain which part of the sentence played a major role in judging hate expressions. If the example sentence is not hate speech, nothing is appeared because the bot judged that it is not hate speech. On the other hand, in the case of the hate speech sentence, our bot prints the visualized image that highlights the key words. For instance, if you use the word "I hate you", the bot outputs an image by highlighting the word "hate" in the sentence.

In addition, it also contains a visualization at the document level. Multiple sentences entered by the user at once can be called a document. Among these sentences, our bot can find which one is played a more important role in judging hate speech. For example, if user types two sentences such as "Everyone loves you. But I hate you.", the bot prints an image by highlighting the sentence "But I hate you." In short, our bot can provide not only finding important words in sentences, but also gives the important sentences in documents. It is possible because of the feature of the hierarchical attention network (HAN) that we used for training the hate speech detection bot.

### Dependencies

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Keras (optional, if you want to use the Keras callback)
- matplotlib (optional, to send convergence plots)

Tested in the following environment:

- Python 3.7
- Tensorflow 1.11
- Keras 2.2.4
- Mac OS

### Installation

1. Clone this repository  
   `git clone https://github.com/smile-speech/hatespeechHAN-telegram-bot.git`

2. Add your token to config.py

   ```python
   # config.py
   TELEGRAM_BOT_TOKEN  = "TOKEN"  # replace TOKEN with your bot's token

   # .gitignore
   config.py
   ```

3. Execute the hatespeechHAN-telegram-bot

   `python hatespeech_bot.py`

### Usage

```python
# Telegram Bot imports
from detection_model import sentiment_analysis

telegram_token = config.TELEGRAM_BOT_TOKEN  # replace TOKEN with your bot's token

# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = config.TELEGRAM_USER_ID   # replace None with your telegram user id (integer):

# Create a sentiment_analysis instance
bot = DLBot(token=telegram_token, user_id=telegram_user_id)

# Activate the bot
bot.activate_bot()
```

### Examples

To create a Telegram bot using the Telegram app, follow these steps:

1. Open the Telegram app
2. Search for the BotFather user (@botfather):

   <img width="409" alt="그림1" src="https://user-images.githubusercontent.com/53829167/95413397-45f60f80-0966-11eb-890a-82e0a6655029.png">

3. Start a conversation with BotFather and click on `start`

4. Send /newbot and follow instructions on screen:

   <img width="327" alt="그림2" src="https://user-images.githubusercontent.com/53829167/95407640-8fd7f900-0958-11eb-9933-d9e235ca7abf.png">

5. Copy the bot token, you will need it when using the DL-Bot:

   <img width="414" alt="그림3" src="https://user-images.githubusercontent.com/53829167/95407701-b7c75c80-0958-11eb-8b22-7e44fc62d341.png">

##### Finding your Telegram user id:

1. Open the Telegram app
2. Search for the userinfobot user (@smilespeech):

   <img width="409" alt="그림4" src="https://user-images.githubusercontent.com/53829167/95408618-d4649400-095a-11eb-9574-57af4db472c4.png">

3. Start a conversation with the bot and get your user id

### References

- [Telegram bots documentation](https://core.telegram.org/bots)
