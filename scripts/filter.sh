#!/usr/bin/env bash

# USE the 3-th GPU

#make clean
#make

# nvprof -m gld_transactions -m gst_transactions ./GSI.exe /home/data/DATASET/gowalla/loc-gowalla_edges.g /home/data/DATASET/gowalla/query/q0.g >& gowalla.prof

# TO find out which one fails, using grep -r "match used" xxx.log
# To count using grep -r "match used" xxx.log |wc -l

data=/home/data/DATASET/

# To run delaunay_n13
query=/home/data/DATASET/query4de/q0.g
 #echo $query
 file=${query##*/}
 # we run the same query 3-times and select the final trial
 ./GSI.exe /home/data/DATASET/delaunay_n13.g ${query} ${result1}${file%.*}.txt 0 >& delaunay_n13.log
echo "delaunay_n13 ends"

# To run enron
query=/home/data/DATASET/query4en/q0.g
    file=${query##*/}
    ./GSI.exe /home/data/DATASET/enron.g ${query} ${result2}${file%.*}.txt 0 >& enron.log
echo "enron ends"

# To run gowalla
query=/home/data/DATASET/query4go/q0.g
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/gowalla.g ${query} ${result3}${file%.*}.txt 0 >& gowalla.log
echo "gowalla ends"
 
# To run road_central
query=/home/data/DATASET/query4ro/q0.g
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/road_central.g ${query} ${result4}${file%.*}.txt 0 >& road_central.log
echo "road_central ends"

# To run watdiv100M
query=/home/data/DATASET/query4wat100/q0.g
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv100M.g ${query} ${result5}${file%.*}.txt 0 >& watdiv100M.log
echo "watdiv100M ends"

# To run dbpedia170M
query=/home/data/DATASET/query4db/q0.g
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/dbpedia170M.g ${query} ${result6}${file%.*}.txt 0 >& dbpedia170M.log
echo "dbpedia170M ends"

