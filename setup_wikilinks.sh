#----- Download Wikilinks corpus

#mkdir data/wikilinks
#mkdir data/wikilinks/source
#cd data/wikilinks/source
#for (( i=1; i<110; i++)) do 
#    f=`printf "%03d" $i` ; 
#    if [ ! -f "$f.gz" ] ; then 
#        echo "Downloading file $i of 109"; 
#        wget http://iesl.cs.umass.edu/downloads/wiki-link/context-only/$f.gz ; 
#    fi
#done ; 
#echo "Downloaded all files, verifying MD5 checksums (might take some time)" ; 
#diff --brief <(wget -q -O - http://iesl.cs.umass.edu/downloads/wiki-link/context-only/md5sum) <(md5sum *.gz) ; 
#if [ $? -eq 1 ] ; then 
#    echo "ERROR: Download incorrect\!" ; 
#    exit 1
#else 
#    echo "Download correct" ; 
#fi

# ------- Extract wikilinks dataset from thrift to json files
# ???

# ------- Rest of the processing is via a python script
python src/prepare_data.py $1 $2 $3 $4
