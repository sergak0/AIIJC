import telebot
import requests
from telebot import types

bot = telebot.TeleBot('1951660616:AAG9jWqi8Hziy_uI5svNDaYd5F9b7GXguws')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
	if message.text == "/start":
		bot.send_message(message.from_user.id, "Здравствуйте! Я бот команды AIIJC \"Dungeon Masters\". Введите текст и следуйте инструкциям, чтобы перевести его в другую категорию")
	else:
		categories = ["Животные", "Музыка", "Спорт", "Литература"]
		keyboard = types.InlineKeyboardMarkup()
		for ind, category in enumerate(categories):
			callback_button = types.InlineKeyboardButton(text=category, callback_data=str(ind)+message.text)
			keyboard.add(callback_button)
		bot.send_message(message.chat.id, "Ваш текст:\n\n" + message.text + "\n\nВыберете категорию этого текста", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def buttons(call):
	if call.message:
		if call.data[0] == 'e':
			categories_eng = ["animals", "music", "sport", "literature"]
			categories = ["Животные", "Музыка", "Спорт", "Литература"]
			langs = ["Русский", "English"]
			langs_eng = ["ru", "en"]
			req = requests.post('http://194.163.165.205:1212' + '/convert_text', json = {'text':call.data[4:], 'original_topic':categories_eng[int(call.data[1])], 'topic':categories_eng[int(call.data[2])], 'language':langs_eng[int(call.data[3])]})
			bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="Ваш текст:\n\n" + call.data[4:] + "\n\nКатегория текста: " + categories[int(call.data[1])] + "\n\nКатегория перевода: " + categories[int(call.data[2])] + "\n\nЯзык текста: " + langs[int(call.data[3])] + "\n\nПереведённый текст:\n\n" + req.json()['text'])
		elif call.data[0] == 'l':
			categories = ["Животные", "Музыка", "Спорт", "Литература"]
			keyboard = types.InlineKeyboardMarkup()
			langs = ["Русский", "English"]
			for ind, lang in enumerate(langs):
				callback_button = types.InlineKeyboardButton(text=lang, callback_data="e"+call.data[1]+call.data[2]+str(ind)+call.data[3:])
				keyboard.add(callback_button)
			bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="Ваш текст:\n\n" + call.data[3:] + "\n\nКатегория текста: " + categories[int(call.data[1])] + "\n\nКатегория перевода: " + categories[int(call.data[2])] + "\n\nВыберите язык Вашего текста", reply_markup=keyboard)
		else:
			categories = ["Животные", "Музыка", "Спорт", "Литература"]
			keyboard = types.InlineKeyboardMarkup()
			for ind, category in enumerate(categories):
				if str(ind) == call.data[0]:
					continue
				callback_button = types.InlineKeyboardButton(text=category, callback_data="l"+call.data[0]+str(ind)+call.data[1:])
				keyboard.add(callback_button)
			bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="Ваш текст:\n\n" + call.data[1:] + "\n\nКатегория текста: " + categories[int(call.data[0])] + "\n\nВыберете категорию, в которую перевести данный текст", reply_markup=keyboard)
		

if __name__ == '__main__':
	bot.polling(none_stop=True, interval=0)