import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import AdamW, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd
import io
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import AdamW, BertForSequenceClassification
import string

class BERT_train_data():
	def __init__(self, tokenizer):
		self.alphabet = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя ')
		self.tokenizer = tokenizer

	def generate_texts(self, texts, windowLen, only_long=True): # Удаление ненужных символов из текста, приведение к нижнему регистру, разбитие текста на более маленькие
		out = []
		for text in texts:
			text = text.lower().replace('\n', ' ')
			text = ''.join(c for c in text if c in self.alphabet)
			text = text.split()
			if only_long:
				out.extend([' '.join(text[x:x+windowLen]) for x in range(0, len(text)-windowLen, windowLen)])
			else:
				out.extend([' '.join(text[x:x+windowLen]) for x in range(0, len(text), windowLen)])
		return out

	def train_data_prepare(self, train_on_facts=True, facts_file_name='facts.csv', train_on_tasks=True, tasks_file_name='full_marked_dataset.csv', sentMaxLen=50, max_len_tokenized=200, batch_size=20, valid_size=0.1):
		
		# Открытие csv файлов и их объединение
		if train_on_facts:
			df = pd.read_csv(facts_file_name)
			df_new = df
		if train_on_tasks:
			df = pd.read_csv(tasks_file_name)
			df = df.drop('Unnamed: 0', 1)
			df = df.rename(columns={'text': 'texts', 'ans': 'labels'})
			df.labels = df.labels.replace('животные', 0).replace('музыка', 1).replace('спорт', 2).replace('литература', 3).replace('неизвестно', 4)
			df = df.drop([i for i in range(len(df.labels)) if df.labels[i] == 4], 0)
			df = df.reset_index(drop=True)
		if train_on_facts and train_on_tasks:
			for i in range(len(df_new)):
			  df.loc[-i-1] = df_new.loc[i]
			df = df.reset_index(drop=True)

		# Формирование тренировочной выборки, её токенизация и преобразование в DataLoader

		sentences = []
		labels = []

		for i in range(len(df.texts)):
			cat_texts = self.generate_texts([df.texts[i]], sentMaxLen, only_long=False)
			label = df.labels[i]
			for j in cat_texts:
				sentences.append(j)
				labels.append(label)
		 
		sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
		X_train = np.array(sentences)
		Y_train = np.array(labels)

		tokenized_texts = [self.tokenizer.tokenize(sent) for sent in X_train]

		tokenized_texts = pad_sequences(
		tokenized_texts,
			maxlen=max_len_tokenized,
			dtype=object,
			truncating="post",
			padding="post",
			value='[PAD]'
		)
		 
		attention_masks = [[1 if i != '[PAD]' else 0 for i in seq] for seq in tokenized_texts]
 
		input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

		train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, Y_train, random_state=42, test_size=valid_size)

		train_masks, validation_masks, _, _ = train_test_split(attention_masks, Y_train, random_state=42, test_size=valid_size)

		train_inputs = torch.tensor(train_inputs)
		train_labels = torch.tensor(train_labels)
		train_masks = torch.tensor(train_masks)

		train_data = TensorDataset(train_inputs, train_masks, train_labels)
		train_dataloader = DataLoader(
			train_data,
			sampler=RandomSampler(train_data),
			batch_size=batch_size
		)

		return train_dataloader, (validation_inputs, validation_labels, validation_masks)




class BERT_model():
	def __init__(self, tokenizer, load_model=True, model_file_name='model.pth', device=torch.device('cpu'), lr=2e-5):
		self.device = device

		# Загрузка модели или создание новой

		self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=4, output_hidden_states=True, output_attentions=True)
		if load_model:
			self.model.load_state_dict(torch.load(model_file_name))
		self.model.to(self.device)

		param_optimizer = list(self.model.named_parameters())
		no_decay = ['bias', 'gamma', 'beta']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
				'weight_decay_rate': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
				'weight_decay_rate': 0.0}
		]

		self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

		self.tokenizer = tokenizer

		self.categories = ("животные", "музыка", "спорт", "литература")
		self.alphabet = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя ')

	def train(self, train_dataloader, validation, epochs=3):
		for ep in range(epochs):
			print(f'Epoch {ep+1}/{epochs}')
			self.model.train()

			# Тренировка модели

			for step, batch in enumerate(train_dataloader):
				batch = tuple(t.to(self.device) for t in batch)
				b_input_ids, b_input_mask, b_labels = batch

				self.optimizer.zero_grad()

				loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
				loss[0].backward()
				self.optimizer.step()

			self.model.eval()
 			
			# Валидация модели

			valid_preds, valid_labels = [], []
			
			for i in range(len(validation[0])):
				b_input_ids = torch.tensor(np.array([validation[0][i]])).to(self.device)
				b_labels = torch.tensor(validation[1][i]).to(self.device)
				b_input_mask = torch.tensor(np.array([validation[2][i]])).to(self.device)

				with torch.no_grad():
					logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

				logits = logits[0].detach().to('cpu').numpy()  
				preds = np.argmax(logits, axis=1).item()
				valid_preds.append(preds)
				valid_labels.append(b_labels.to('cpu').numpy().item())

			print("Accuracy на валидационной выборке: {0:.2f}%".format(
			    accuracy_score(valid_labels, valid_preds) * 100
			))
			print("f1: {0:.2f}%".format(
			    f1_score(valid_labels, valid_preds, average='macro') * 100
			))

	def prediction(self, sentence, max_key_words=1): # Получение категории и ключевых слов текста
		# Подготовка текста: удаление ненужных символов из текста, приведение к нижнему регистру, токенизация
		sentence = sentence.lower().replace('\n', ' ').replace('ё', 'е').replace('й', 'и')
		sentence = ''.join(c for c in sentence if c in self.alphabet)
		sentence = ' '.join(sentence.split()[:200])
		sentence = '[CLS] ' + sentence + ' [SEP]'
		words = sentence.split()
		tokenized = self.tokenizer.tokenize(sentence)

		# Получение для каждого токена индекса слова, из которого сделан этот токен
		now_word = 0
		now_ind = 0
		token_to_word = []
		for i in tokenized:
			token_to_word.append(now_word)
			# Проход по всем символам токена
			for j in range(len(i)):
				if i[j] != words[now_word][now_ind]:
					continue
				now_ind += 1
				if now_ind == len(words[now_word]):
					now_ind = 0
					now_word += 1

		# Предсказание модели и получение attention
		ids = [self.tokenizer.convert_tokens_to_ids(tokenized)]
		ids = torch.tensor(ids)
		ids = ids.to(self.device)
		label = torch.tensor([0]).to(self.device)
		with torch.no_grad():
			logits = self.model(ids, token_type_ids=None, attention_mask=None, labels=label)

		ans = logits[1].detach().to('cpu').numpy()
		ans_ind = np.argmax(ans, axis=1).item()
		ans_cat = self.categories[ans_ind]

		attention = logits[3][11][0][11][0].detach().to('cpu').numpy()

		# Получение самых важных max_key_words слов
		max_inds = np.flip(np.argsort(attention))
		all_key_words = ['[SEP]', '[CLS]']
		for i in max_inds:
			# Просмотр новых слов до получения max_key_words самых важных слов или до просмотра всех слов
			if words[token_to_word[i]] in all_key_words:
				continue
			all_key_words.append(words[token_to_word[i]])
			if len(all_key_words)-2 == max_key_words:
				break
		return ans_cat, all_key_words[2:]

	def save_model(self, model_file_name='model_new.pth'):
		torch.save(self.model.state_dict(), model_file_name)

	def print_one_test(self, sentence, max_key_words=1):
		ans_cat, all_key_words = self.prediction(sentence, max_key_words)
		print(ans_cat)
		print(all_key_words)

	def test(self, test_file_name='test.csv', ans_file_name='ans.csv'):
		# Создание предсказаний модели на тестовой выборке -- ans.csv
		self.model.eval()
		df_test = pd.read_csv(test_file_name)
		ans_csv = []
		for i, text in enumerate(df_test.task):
			ans_cat, all_key_words = self.prediction(text, 3)
			ans_csv.append([i, ans_cat, '; '.join(all_key_words)])
		ans_df = pd.DataFrame(ans_csv, columns=['id', 'category', 'keywords'])
		ans_df.to_csv(ans_file_name, index=False)




if __name__ == '__main__':
	use_cpu = False

	if use_cpu:
		device = torch.device("cpu")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	trained = True
	tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=True)
	bert = BERT_model(tokenizer, load_model=trained, device=device)
	if not trained:
		train_data = BERT_train_data(tokenizer)
		train_dataloader, validation = train_data.train_data_prepare()
		bert.train(train_dataloader, validation)
		bert.save_model()
	
	bert.test()

"""
Наш github, в котором находится информация о всей проделанной нами работе: https://github.com/sergak0/AIIJC
Наш telegram-bot: @aiijcdungeonmastersbot
"""
