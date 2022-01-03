from telegram.ext import (Updater,MessageHandler,Filters)
import numpy as np
import logging 
import config
import seaborn as sn
from detection_model import sentiment_analysis
from io import BytesIO
from matplotlib import pyplot as plt
plt.switch_backend('Agg')
logger = logging.getLogger(__name__)


class DLBot(object):
    def __init__(self, token, user_id=None):
        assert isinstance(token, str), 'Token must be of type string'
        self.token = token  
        self.user_id = user_id  
        self.loss_hist = []
        self.word_list = []
        self.logger = logging.getLogger(__name__)
        self.warning_message = "WARNING"


    def activate_bot(self):
        self.updater = Updater(config.TELEGRAM_BOT_TOKEN,use_context = True)
        dp = self.updater.dispatcher  
        dp.add_handler(MessageHandler(Filters.text, self.plot_loss ))
        self.updater.start_polling()
        self.bot_active = True


    def plot_loss(self,update,context):
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
                fig.subplots_adjust(wspace=0.1)
                heatmap = sn.heatmap([[sent_att[0][sent_idx]]], xticklabels=False, yticklabels=False,ax=ax[sent_idx,0],cbar = False , annot=[[sent_att_labels[0][sent_idx]]],annot_kws={"size": 35,"color":"k"} ,fmt ='', square=True, linewidths=0,center=0.3,vmax=1, cmap='Blues')
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
    telegram_user_id = config.TELEGRAM_USER_ID   
    bot = DLBot(token=telegram_token, user_id=telegram_user_id)
    bot.activate_bot()

if __name__ == '__main__':
    main()
