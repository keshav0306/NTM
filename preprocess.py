# file_en = "es-en/europarl-v7.es-en.en"
# file_es = "es-en/europarl-v7.es-en.es"

file_en = "english_million.txt"
file_es = "spanish_million.txt"

num = 100000

# for i in range(num):
#     file3.writelines(file1.readline())


##########################################
# Source : https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/

import string
import re
from pickle import dump
from unicodedata import normalize

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	# dump(sentences, open(filename, 'wb'))
    file = open(filename, "w")
    sentences = [sen + "\n" for sen in sentences]
    file.writelines(sentences)
	# print('Saved: %s' % filename)

# load English data

doc = load_doc(file_en)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'english.txt')
# spot check
for i in range(10):
	print(sentences[i])

# load French data
filename = file_es
doc = load_doc(filename)
sentences = to_sentences(doc)
sentences = clean_lines(sentences)
save_clean_sentences(sentences, 'spanish.txt')
# spot check
for i in range(10):
	print(sentences[i])