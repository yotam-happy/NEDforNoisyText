mkdir data/intralinks
mkdir data/intralinks/all_tmp
mkdir data/intralinks/all

python src/wikixml.py intralink_jsons data/enwiki/extracted data/intralinks/all $1 $2 $3 $4

mkdir data/PPRforNED
cd data/PPRforNED
wget -O PPRforNED.zip "https://github.com/masha-p/PPRforNED/archive/master.zip"
unzip PPRforNED.zip
cd ../..

python src/WikilinksStatistics.py $1 $2 $3 $4

