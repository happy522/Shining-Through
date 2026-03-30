# Shining-Through
Implemetation with different dataset for finding shining-through effect when source language text is translated in English.

[Europarl:](https://www.statmt.org/europarl/)
- We have used dataset Version 7 [Download](https://www.statmt.org/europarl/v7/europarl.tgz)
- Data extraction: We have kept only those files where source language are mentioned as: DE, FR, ES and IT and filtered out using [this script](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl_data_extractor.py). It maps the English translation to it's source language, by matching files names and creates [source_language]_output.csv file. 
- Conllu files: construct conllu files using [this script](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl_conllu_generation.py) after extracting the data as mentioned above. Constructred conllu files can be found in zip file [here](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl.zip)
- Feature construction for Shining through: Use [this](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/mega_collector.py) - change the name of generated conllu file accordingly to create features that can be used for further implementation.
- Implementation: [Script:](https://github.com/happy522/Shining-Through/blob/main/Implementation/europal/europarl_shining_through.ipynb) It can run without above steps as data after all these steps is available in Github [here](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/europarl/europarl_all_data.csv) We have merged all DE, ES, FR and IT data in one file for implementation.

[Ted-Talk:] https://www.openslr.org/100/
- We have used aligned translations for FR, ES and IT.
- Download the dataset, remove wav folders from each.
- [Run](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/ted-talk/merge_files.py) for merging the files in one and keeping source language name, source and translation texts.
- [Run](https://github.com/happy522/Shining-Through/blob/main/Data%20Generation/Datasets/ted-talk/ted-talk-conllu_generation.py) for individual conllu folders generation.
- Note: We are searching for German aligned translations.

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
│   │   │   └── europarl_data_extractor.py
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
