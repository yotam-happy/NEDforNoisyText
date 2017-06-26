#---- get other projects

mkdir word2vecf
cd word2vecf
wget -O word2vecf.zip "https://bitbucket.org/yoavgo/word2vecf/get/1b94252a58d4.zip"
unzip word2vecf.zip
cd yoavgo-word2vecf-1b94252a58d4
make
cd ../..

mkdir wikiextractor
cd wikiextractor
wget -O wikiextractor.zip "https://github.com/attardi/wikiextractor/zipball/master"
unzip wikiextractor.zip
cd ..

mkdir WikilinkIterator2JSON
cd WikilinkIterator2JSON
wget -O WikilinkIterator2JSON.zip "https://github.com/NoamGit/wiki2JSON/archive/master.zip"
unzip WikilinkIterator2JSON.zip
cd ..

#------ extract all documents from wiki dump

if [ ! -f "data/enwiki/extracted" ] ; then 
    mkdir data/enwiki/extracted
    cd wikiextractor/attardi-wikiextractor-2a5e6ae
    python WikiExtractor.py -o ../../data/enwiki/extracted -l -b 100M --processes 3 -q ../../data/enwiki/pages-articles.xml 
    cd ../..
else 
    echo "couldnt extact wiki xml content";
    exit 1
fi

#------ generate training dataset for word2vecf
mkdir data/word2vec
python src/wikixml.py w2v_train data/enwiki/extracted data/word2vec/training_corpus.txt

#------ Train word&entity vectors
word2vecf/yoavgo-word2vecf-1b94252a58d4/count_and_filter -train data/word2vec/training_corpus.txt -cvocab data/word2vec/cv -wvocab data/word2vec/wv -min-count 20
word2vecf/yoavgo-word2vecf-1b94252a58d4/word2vecf -train data/word2vec/training_corpus.txt -wvocab data/word2vec/wv -cvocab data/word2vec/cv -output data/word2vec/dim300vecs -size 300 -negative 15 -threads 10 -dumpcv data/word2vec/dim300context-vecs -iters 10

