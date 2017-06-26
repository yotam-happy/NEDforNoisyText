# NEDforNoisyText
Named Entity Disambiguation for Noisy Text

This repository contains code for reproducing our experiments in our "Named Entity Disambiguation for Noisy Text" submittion to CoNLL 2017.
Some manual steps are required to setup the data for the experiments

- Please setup a mysql schema with the page and redirect tables from a Wikipedia dump.
- Please place the wikipedia pages-article xml file at data/enwiki/pages-articles.xml.
- For processing wikilinks files from umass an installationg of scala is required.

python libraries required for the project: keras, tensorflow, numpy, nltk, json, unicodedata, unidecode, mysql.connector, urllib, matplotlib, zipfile, ujson, pandas, urlparse, sklearn.

For running the CoNLL test

- Create a data/CoNLL folder and place the aida-conll dataset inside (CoNLL_AIDA-YAGO2-dataset.tsv)
To generate this dataset refer to https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/


Seting up the data
run ./setup_data.sh
to setup data for the wikilinksNED evaluation run ./setup_wikilinks.sh <db_user> <db_pass> <db_schema> <db_ip>
to setup data for the conll evaluation run ./setup_conll.sh <db_user> <db_pass> <db_schema> <db_ip>

REMARK: as of now you are required to use the script at https://github.com/NoamGit/Wiki2JSON/ to extract Wikilinks from its original format (thrift) to an easier to work with json format. The resulting files should be stored at data/wikilinks/unprocessed. We are working to automate this process as soon as possible.

Running evaluations
for running WikilinksNED evaluation run ./evaluateWikilinksNED.sh <db_user> <db_pass> <db_schema> <db_ip>
for running CoNLL evaluation run ./pretrainOnWikipedia.sh <db_user> <db_pass> <db_schema> <db_ip>
and then ./evaluateCoNLL.sh <db_user> <db_pass> <db_schema> <db_ip>


results are written to evaluation.txt file in the main directory

NOTES:
- Setting up data and running experiments takes a very long time.
- After setting up data for both experiments the data folder can reach 300+Gb

This code is provided as-is. Running this code or any part of it is at your own risk. We do not take any responsibility for running any of the code or usage of any of the data. 
Much of this code was written at the same time as I was learning Keras, TF and Theano. It has undergone many changes and was used for extensive experimentation. It is therefore probably full of design flaws and redundancies. 
