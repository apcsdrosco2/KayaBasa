from pathlib import Path
import sys
from TRAD import *
from SYLL import *
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
    output_path = output_dir / f"{lang}_syll.csv"
    label, title = [], []
    cc, v, cv, vc, cvc, vcc, cvcc, ccvc, ccv, ccvcc, ccvccc = [], [], [], [], [], [], [], [], [], [], []

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as file:
        for item in file.readlines():
            parsed = item.split(',',2)
            if len(parsed) < 3 or not parsed[2].strip(): continue
            text = parsed[2].strip()

            title.append(parsed[0])
            label.append(parsed[1])
            cc.append(get_consonant_cluster(text)); v.append(get_v(text))
            cv.append(get_cv(text)); vc.append(get_vc(text))
            cvc.append(get_cvc(text)); vcc.append(get_vcc(text))
            cvcc.append(get_cvcc(text)); ccv.append(get_ccv(text))
            ccvc.append(get_ccvc(text)); ccvcc.append(get_ccvcc(text))
            ccvccc.append(get_ccvccc(text))

    df = pd.DataFrame(list(zip(title, cc, v, cv, vc, cvc, vcc, cvcc, ccvc, ccv, ccvcc, ccvccc, label)),
                      columns=['book_title', 'consonant_cluster_density', 'v_density', 'cv_density', 'vc_density',
                               'cvc_density','vcc_density','cvcc_density','ccvc_density','ccv_density','ccvcc_density','ccvccc_density','grade_level'])
    df.to_csv(output_path, index=False)
    print(f'{lang.upper()} SYLL DONE')