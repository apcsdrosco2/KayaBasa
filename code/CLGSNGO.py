from pathlib import Path
import re

NGRAMS_DIR = Path(__file__).resolve().parent.parent / 'data-cleaning-repo-main' / 'ngrams list'

def clean(text):
  text = re.sub('[^A-Za-z ]',' ',text)
  text = re.sub(' +',' ',text)
  text = text.lower()
  return text

def get_ngrams(s, n):
  n_gram_list = [s[i:i+n] for i in range(len(s)-n+1)]
  n_gram_list_filtered = [] #remove ngrams with spaces
  for i in n_gram_list:
    if " " not in i:
      n_gram_list_filtered.append(i)
  return n_gram_list_filtered

def get_bigram_CLGSNGO(text):
  text = clean(text)
  input_text_ngram = get_ngrams(text,2)
  input_text_ngram_count = len(input_text_ngram)

  bigram_tag_counter = 0
  bigram_ceb_counter = 0
  bigram_bik_counter = 0

  bigram_tag_match = 0
  bigram_bik_match = 0
  bigram_ceb_match = 0

  bigram_tag_loc = NGRAMS_DIR / "tag_top25_2gram.txt"
  bigram_ceb_loc = NGRAMS_DIR / "ceb_top25_2gram.txt"
  bigram_bik_loc = NGRAMS_DIR / "bik_top25_2gram.txt"

  bigram_tag_list = []
  bigram_ceb_list = []
  bigram_bik_list = []

  with open(bigram_tag_loc,'r') as bigram_tag_file:
    bigram_tag_file_contents = bigram_tag_file.readlines()
    for i in bigram_tag_file_contents:
      item = i.strip().split(",")
      bigram_tag_list.append(item[0])

  with open(bigram_ceb_loc,'r') as bigram_ceb_file:
    bigram_ceb_file_contents = bigram_ceb_file.readlines()
    for i in bigram_ceb_file_contents:
      item = i.strip().split(",")
      bigram_ceb_list.append(item[0])

  with open(bigram_bik_loc,'r') as bigram_bik_file:
    bigram_bik_file_contents = bigram_bik_file.readlines()
    for i in bigram_bik_file_contents:
      item = i.strip().split(",")
      bigram_bik_list.append(item[0])

  for i in input_text_ngram:
    if i in bigram_tag_list:
      bigram_tag_counter += 1
    if i in bigram_bik_list:
      bigram_bik_counter += 1
    if i in bigram_ceb_list:
      bigram_ceb_counter += 1

  if bigram_tag_counter != 0:
    bigram_tag_match = bigram_tag_counter/input_text_ngram_count
  if bigram_bik_counter != 0:
    bigram_bik_match = bigram_bik_counter/input_text_ngram_count
  if bigram_ceb_counter != 0:
    bigram_ceb_match = bigram_ceb_counter/input_text_ngram_count

  return bigram_tag_match,bigram_bik_match,bigram_ceb_match


def get_trigram_CLGSNGO(text):
  text = clean(text)
  input_text_ngram = get_ngrams(text,3)
  input_text_ngram_count = len(input_text_ngram)

  trigram_tag_counter = 0
  trigram_ceb_counter = 0
  trigram_bik_counter = 0

  trigram_tag_match = 0
  trigram_bik_match = 0
  trigram_ceb_match = 0

  trigram_tag_loc = NGRAMS_DIR / "tag_top25_3gram.txt"
  trigram_ceb_loc = NGRAMS_DIR / "ceb_top25_3gram.txt"
  trigram_bik_loc = NGRAMS_DIR / "bik_top25_3gram.txt"

  trigram_tag_list = []
  trigram_ceb_list = []
  trigram_bik_list = []

  with open(trigram_tag_loc,'r') as trigram_tag_file:
    trigram_tag_file_contents = trigram_tag_file.readlines()
    for i in trigram_tag_file_contents:
      item = i.strip().split(",")
      trigram_tag_list.append(item[0])

  with open(trigram_ceb_loc,'r') as trigram_ceb_file:
    trigram_ceb_file_contents = trigram_ceb_file.readlines()
    for i in trigram_ceb_file_contents:
      item = i.strip().split(",")
      trigram_ceb_list.append(item[0])

  with open(trigram_bik_loc,'r') as trigram_bik_file:
    trigram_bik_file_contents = trigram_bik_file.readlines()
    for i in trigram_bik_file_contents:
      item = i.strip().split(",")
      trigram_bik_list.append(item[0])

  for i in input_text_ngram:
    if i in trigram_tag_list:
      trigram_tag_counter += 1
    if i in trigram_bik_list:
      trigram_bik_counter += 1
    if i in trigram_ceb_list:
      trigram_ceb_counter += 1

  if trigram_tag_counter != 0:
    trigram_tag_match = trigram_tag_counter/input_text_ngram_count
  if trigram_bik_counter != 0:
    trigram_bik_match = trigram_bik_counter/input_text_ngram_count
  if trigram_ceb_counter != 0:
    trigram_ceb_match = trigram_ceb_counter/input_text_ngram_count

  return trigram_tag_match,trigram_bik_match,trigram_ceb_match