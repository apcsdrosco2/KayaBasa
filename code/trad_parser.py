from pathlib import Path
import sys
from TRAD import *
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
clean_folder = project_dir / 'clean'
output_dir = Path(__file__).resolve().parent / 'generated'
output_dir.mkdir(parents=True, exist_ok=True)

files = ['bik_all_clean.txt', 'ceb_all_clean.txt', 'tag_all_clean.txt']

for file_name in files:
    input_path = clean_folder / file_name
    if not input_path.exists(): continue

    lang = file_name.split('_')[0]
    output_path = output_dir / f"{lang}_trad.csv"
    label, title = [], []
    wc, sc, pc, awl, asl, awsc, polysyll = [], [], [], [], [], [], []

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as file:
        for item in file.readlines():
            parsed = item.split(',',2)
            if len(parsed) < 3 or not parsed[2].strip(): continue
            text = parsed[2].strip()

            title.append(parsed[0])
            label.append(parsed[1])
            wc.append(word_count_per_doc(text)); sc.append(sentence_count_per_doc(text))
            pc.append(ave_phrase_count_per_doc(text)); awl.append(ave_word_length(text))
            asl.append(word_count_per_sentence(text)); awsc.append(ave_syllable_count_of_word(text))
            polysyll.append(polysyll_count_per_doc(text))

    df = pd.DataFrame(list(zip(title, wc, sc, pc, awl, asl, awsc, polysyll, label)),
                      columns=['book_title','word_count','sentence_count', 'phrase_count_per_sentence', 
                               'average_word_len', 'average_sentence_len', 'average_syllable_count', 'polysyll_count', 'grade_level'])
    df.to_csv(output_path, index=False)
    print(f'{lang.upper()} TRAD DONE')