from pathlib import Path
import sys

from TRAD import *
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_dir / 'clean' / 'ceb_all_clean.txt'
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / 'trad.csv'

with open(input_path, 'r', encoding='utf-8', errors='ignore') as file:
	file_contents = file.readlines()

	label = []
	title = []
	word_count = []
	sentence_count = []
	phrase_count = []
	average_word_length = []
	average_sentence_length = []
	average_word_syll_count = []
	polysyll_count = []

	for item in file_contents:
		parsed_text = item.split(',',2)
		if len(parsed_text) < 3 or not parsed_text[2].strip():
			continue
		parsed_text[2] = parsed_text[2].strip()

		print(parsed_text[0], parsed_text[1])

		title.append(parsed_text[0])
		label.append(parsed_text[1])
		word_count.append(word_count_per_doc(parsed_text[2]))
		sentence_count.append(sentence_count_per_doc(parsed_text[2]))
		phrase_count.append(ave_phrase_count_per_doc(parsed_text[2]))
		average_word_length.append(ave_word_length(parsed_text[2]))
		average_sentence_length.append(word_count_per_sentence(parsed_text[2]))
		average_word_syll_count.append(ave_syllable_count_of_word(parsed_text[2]))
		polysyll_count.append(polysyll_count_per_doc(parsed_text[2]))

df = pd.DataFrame(list(zip(title, word_count, sentence_count, phrase_count, average_word_length, average_sentence_length, average_word_syll_count, polysyll_count, label)),columns=['book_title','word_count','sentence_count', 'phrase_count_per_sentence', 'average_word_len', 'average_sentence_len', 'average_syllable_count', 'polysyll_count', 'grade_level'])

df.to_csv(output_path, index=False)

print('\nFEATURE EXTRACTION DONE')