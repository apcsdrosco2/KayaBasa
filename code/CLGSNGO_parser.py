from pathlib import Path
import sys
from CLGSNGO import *
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
    output_path = output_dir / f"{lang}_clgsngo.csv"
    label, title = [], []
    tag_bi, bik_bi, ceb_bi, tag_tri, bik_tri, ceb_tri = [], [], [], [], [], []

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as file:
        for item in file.readlines():
            parsed = item.split(',',2)
            if len(parsed) < 3 or not parsed[2].strip(): continue
            
            title.append(parsed[0])
            label.append(parsed[1])
            
            t_b, b_b, c_b = get_bigram_CLGSNGO(parsed[2].strip())
            tag_bi.append(t_b); bik_bi.append(b_b); ceb_bi.append(c_b)

            t_t, b_t, c_t = get_trigram_CLGSNGO(parsed[2].strip())
            tag_tri.append(t_t); bik_tri.append(b_t); ceb_tri.append(c_t)

    df = pd.DataFrame(list(zip(title, tag_bi, bik_bi, ceb_bi, tag_tri, bik_tri, ceb_tri, label)),
                      columns=['book_title','tagalog_bigram_sim','bikol_bigram_sim','cebuano_bigram_sim',
                               'tagalog_trigram_sim','bikol_trigram_sim','cebuano_trigam_sim', 'grade_level'])
    df.to_csv(output_path, index=False)
    print(f'{lang.upper()} CLGSNGO DONE')