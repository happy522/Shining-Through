# Shining-Through
Implemetation with different dataset for finding shining-through effect when source language text is translated in English.


## Project Structure

```
Shining-Through/
├── Data Generation/
│   ├── Datasets/
│   │   ├── L2_TR/
│   │   ├── europarl/
│   │   │   ├── English_Text_and_source_lang.csv
│   │   │   ├── europarl_all_data.csv
│   │   │   ├── europarl_conllu_generation.py
│   │   │   ├── europarl_data_extractor.py
│   │   │   └── extract_en_text_and_source_language.py
│   │   └── ted-talk/
│   ├── searchlists/
│   ├── conllu_generation.py
│   ├── extractors.py
│   ├── helpfunctions.py
│   ├── mega_collector.py
│   ├── our45features_extracted.tsv
│   ├── lrec20_45featureset_description.pdf
│   ├── __init__.py
│   ├── .gitattributes
│   ├── .gitignore
│   └── README.md
├── Implementation/
│   └── L2_TR/
├── TR_L2_shining_through.ipynb
├── europarl_shining_through.ipynb
├── .gitattributes
└── README.md
