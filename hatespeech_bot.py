""" Deep Learning Telegram bot
DLBot and TelegramBotCallback classes for the monitoring and control
of a Keras Tensorflow training process using a Telegram bot
By: Eyal Zakkay, 2019
https://eyalzk.github.io/
"""

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler,MessageHandler, Filters, RegexHandler,
                          ConversationHandler)
#from telegram_bot_callback import TelegramBotCallback

import numpy as np

import logging 
import config


logger = logging.getLogger(__name__)
from io import BytesIO
from matplotlib import pyplot as plt
plt.switch_backend('Agg')

import seaborn as sn

from detection_model import sentiment_analysis



class DLBot(object):
    """  A class for interacting with a Telegram bot to monitor and control a Keras \ tensorflow training process.
    Supports the following commands:
     /start: activate automatic updates every epoch and get a reply with all command options
     /help: get a reply with all command options
     /status: get a reply with the latest epoch's results
     /getlr: get a reply with the current learning rate
     /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
     /plot: get a reply with the loss convergence plot image
     /quiet: stop getting automatic updates each epoch
     /stoptraining: kill training process

    # Arguments
        token: String, a telegram bot token
        user_id: Integer. Specifying a telegram user id will filter all incoming
                 commands to allow access only to a specific user. Optional, though highly recommended.
    """


    def __init__(self, token, user_id=None):
        assert isinstance(token, str), 'Token must be of type string'
        assert user_id is None or isinstance(user_id, int), 'user_id must be of type int (or None)'

        self.token = token  # bot token
        self.user_id = user_id  # id of the user with access
        self.filters = None
        self.chat_id = None  # chat id, will be fetched during /start command
        self.bot_active = False  # currently not in use
        self.updater = None
        # Initialize loss monitoring
        self.loss_hist = []
        self.word_list = []
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Message to display on /start and /help commands
        self.startup_message = "Hi, I'm the DL bot! I will send you updates on your training process.\n" \
                               " send /start to activate automatic updates every epoch\n" \
                               " send /help to see all options.\n"

        self.warning_message = "WARNING"

    def activate_bot(self):
        """ Function to initiate the Telegram bot """
        self.updater = Updater(config.TELEGRAM_BOT_TOKEN,use_context = True)  # setup updater
        dp = self.updater.dispatcher  # Get the dispatcher to register handlers
        #dp.add_error_handler(self.error)  # log all errors
        #self.filters = Filters.user(user_id=self.user_id) if self.user_id else None
        #self.filters = Filters.user(user_id=self.user_id) if self.user_id else None
        # Command and conversation handles
        dp.add_handler(CommandHandler("start", self.start))#, filters=self.filters))  # /start
        dp.add_handler(CommandHandler("help", self.help))#,filters=self.filters))  # /help
        dp.add_handler(MessageHandler(Filters.text, self.plot_loss ))


        # Start the Bot
        self.updater.start_polling()
        self.bot_active = True

    def start(self, update,context):
        """ Telegram bot callback for the /start command.
        Fetches chat_id, activates automatic epoch updates and sends startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id
        self.verbose = True


    def help(self, bot, update):
        """ Telegram bot callback for the /help command. Replies the startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id



    def plot_loss(self,update,context):
        """ Telegram bot callback for the /plot command. Replies with a convergence plot image"""
        check_hate, pred_attention, tokenized_sentences, sent_attention, word_rev_index= sentiment_analysis(update.message.text)

        if check_hate == 0  :
            return
        
        else:
            MAX_SENTENCE_LENGTH = 25

            sent_att_labels=[]
            for sent_idx, sentence in enumerate(tokenized_sentences):
                if sentence[-1] == 0:
                    continue
                sent_len = sent_idx
                sent_att_labels.append("Sent "+str(sent_idx+1))
            sent_att = sent_attention[0:sent_len+1]
            sent_att = np.expand_dims(sent_att, axis=0)
            sent_att_labels = np.expand_dims(sent_att_labels, axis=0)


            fig,ax = plt.subplots(len(sent_att_labels[0]), 2,figsize=(20, 9),gridspec_kw={'width_ratios': [1, 20]},sharex='col',squeeze=False)


            for sent_idx, sentence in enumerate(tokenized_sentences):
                if sentence[-1] == 0:
                    continue
                for word_idx in range(MAX_SENTENCE_LENGTH):
                    if sentence[word_idx] != 0:
                        words = [word_rev_index[word_id] for word_id in sentence[word_idx:]]
                        pred_att = pred_attention[sent_idx][-len(words):]
                        pred_att = np.expand_dims(pred_att, axis=0)
                        break
                word_list = np.expand_dims(words, axis=0)   
        
                

                #fig,(ax, ax2) = plt.subplots(1,2,figsize=(20, 10),gridspec_kw={'width_ratios': [1, len(words)]},sharex='col')
                fig.subplots_adjust(wspace=0.1)
                #plt.rc('xtick', labelsize=10)
                #cmap="Blues",cmap='YlGnBu"
                print(sent_att[0])
                heatmap = sn.heatmap([[sent_att[0][sent_idx]]], xticklabels=False, yticklabels=False,ax=ax[sent_idx,0],cbar = False , annot=[[sent_att_labels[0][sent_idx]]],annot_kws={"size": 35,"color":"k"} ,fmt ='', square=True, linewidths=0,center=0.3,vmax=1, cmap='Blues')
                #plt.xticks(rotation=45)
                #plt.show()
                
                
                #fig, ax = plt.subplots(figsize=(len(words), 2))
                #plt.rc('xtick', labelsize=10)
                word_list = np.expand_dims(words, axis=0) 
                heatmap = sn.heatmap(pred_att, xticklabels=False, yticklabels=False,ax=ax[sent_idx,1],cbar=False, square=True,annot=word_list ,fmt ='', annot_kws={"alpha":1,'rotation':30,"size": 40},cmap ="bwr", linewidths=0, vmin=0,center=0, vmax=0.4)
                plt.xticks(rotation=45)



            plt.show()
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            update.message.reply_text(self.warning_message, reply_markup=ReplyKeyboardRemove())
            update.message.reply_photo(buffer)



def main():

    telegram_token = config.TELEGRAM_BOT_TOKEN  # replace TOKEN with your bot's token

    # user id is optional, however highly recommended as it limits the access to you alone.
    telegram_user_id = config.TELEGRAM_USER_ID   # replace None with your telegram user id (integer):

    # Create a DLBot instance
    bot = DLBot(token=telegram_token, user_id=telegram_user_id)

    # Activate the bot
    bot.activate_bot()
    # Basic model parameters as external flags.
    FLAGS = None


if __name__ == '__main__':
    main()
