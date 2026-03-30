# Shining-Through
This project investigates the **shining-through effect**, where linguistic properties of a **source language** persist in its **English translations**.

We build a full pipeline that:
- Processes multilingual datasets  
- Extracts linguistic features  
- Trains models to detect source-language influence  

---

# Overview
The goal is to determine whether translated English text retains patterns from:
- German (DE)
- French (FR)
- Spanish (ES)
- Italian (IT)

We achieve this by:
1. Converting text into **CONLLU format**
2. Extracting **45 linguistic features**
3. Training models to classify source language

---

# Datasets

## 1. L2 and TR
- Source:  L2:parts of the TOEFL11 corpus (Blanchard et al., 2013) - a dataset of English-as-a-foreign language essays
- TR: scientific publications abstracts of students SciPar corpus (Roussis et al. 2022) - translated from other language to English
- Languages: DE, FR, ES, IT


## 2. Europarl
- Source: https://www.statmt.org/europarl/
- Version: v7
- Languages: DE, FR, ES, IT


## 3. TED Talks
- Source: https://www.openslr.org/100/
- Languages: FR, ES, IT


## 4. Subtitles
- Source: OpenSubtitles API
  
---

# Pipeline
Raw Data → Extraction → CONLLU Generation → Feature Extraction → Model Training


## Europarl: https://www.statmt.org/europarl/
Data Generation:
- We have used dataset Version 7 [Download](https://www.statmt.org/europarl/v7/europarl.tgz)
- Data extraction: We have kept only those files where source language are mentioned as: DE, FR, ES and IT and filtered out using [this script](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl_data_extractor.py). It maps the English translation to it's source language, by matching files names and creates [source_language]_output.csv file. 
- Conllu files: construct conllu files using [this script](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl_conllu_generation.py) after extracting the data as mentioned above. Constructred conllu files can be found in zip file [here](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl.zip)
- Feature construction for Shining through: Use [this script](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/mega_collector.py) - change the name of generated conllu file accordingly to create features that can be used for further implementation.
  
- Implementation: [this script](https://github.com/happy522/Shining-Through/blob/main/Implementation/europal/europarl_shining_through.ipynb) It can run without above steps as data after all these steps is available in Github [here](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl_all_data.csv) We have merged all DE, ES, FR and IT data in one file for implementation.

## Ted-Talk: https://www.openslr.org/100/
Data Generation:
- We have used aligned translations for FR, ES and IT.
- Download the dataset, remove wav folders from each.
- [Run](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/ted-talk/merge_files.py) for merging the files in one and keeping source language name, source and translation texts.
- [Run](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/ted-talk/ted-talk-conllu_generation.py) for individual conllu folders generation.
- Note: We are searching for German aligned translations.

## Subtitles: https://api.opensubtitles.com/api/v1/subtitles
- Using API we can extract the subtitles.
  
---
  
# Project Structure

```
Shining-Through/
├── Data Generation/
│   ├── Datasets/
│   │   ├── L2_TR/
│   │   ├── europarl/
│   │   │   ├── europarl.zip # contains all conllu files of english and source language, features csv file for each language.
│   │   │   ├── europarl_all_data.csv #combined features file for implementation
│   │   │   ├── europarl_conllu_generation.py # For conllu file generation for each english text of source language
│   │   │   └── europarl_data_extractor.py #data extraction from main downloaded folder
│   │   └── ted-talk/
│   │   └── subtitles/
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
│   ├── L2_TR/
│   │   └──TR_L2_shining_through.ipynb
│   └── Europarl/
│   │   └──europarl_shining_through.ipynb
├── .gitattributes
└── README.md
