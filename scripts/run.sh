#!/usr/bin/env bash

# USE the 3-th GPU

#make clean
#make

# nvprof -m gld_transactions -m gst_transactions ./GSI.exe /home/data/DATASET/gowalla/loc-gowalla_edges.g /home/data/DATASET/gowalla/query/q0.g >& gowalla.prof

# TO find out which one fails, using grep -r "match used" xxx.log
# To count using grep -r "match used" xxx.log |wc -l

data=/home/data/DATASET/

# To run delaunay_n13
result1=delaunay_n13.log/
file1=delaunay_n13.tmp
/bin/rm -rf ${result1}
mkdir ${result1}
for query in `ls /home/data/DATASET/query4de/*`
do
 #echo $query
 file=${query##*/}
 # we run the same query 3-times and select the final trial
 ./GSI.exe /home/data/DATASET/delaunay_n13.g ${query} ${result1}${file%.*}.txt 3 >& ${result1}${file%.*}.log
 #./GSI.exe /home/data/DATASET/delaunay_n13.g ${query} ${result1}${file%.*}.txt 3 >& ${result1}${file%.*}.log
 #./GSI.exe /home/data/DATASET/delaunay_n13.g ${query} ${result1}${file%.*}.txt 3 >& ${result1}${file%.*}.log
 grep "match used" ${result1}${file%.*}.log >> ${file1}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering delaunay_n13 queries: %.2f ms\n", t/cnt);}' ${file1}
/bin/rm -f $file1
echo "delaunay_n13 ends"

# To run enron
result2=enron.log/
file2=enron.tmp
/bin/rm -rf ${result2}
mkdir ${result2}
for query in `ls /home/data/DATASET/query4en/*`
do
    file=${query##*/}
    ./GSI.exe /home/data/DATASET/enron.g ${query} ${result2}${file%.*}.txt 3 >& ${result2}${file%.*}.log
    grep "match used" ${result2}${file%.*}.log >> ${file2}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering enron queries: %.2f ms\n", t/cnt);}' ${file2}
/bin/rm -f $file2
echo "enron ends"

# To run gowalla
result3=gowalla.log/
file3=gowalla.tmp
/bin/rm -rf ${result3}
mkdir ${result3}
for query in `ls /home/data/DATASET/query4go/*`
do
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/gowalla.g ${query} ${result3}${file%.*}.txt 3 >& ${result3}${file%.*}.log
 grep "match used" ${result3}${file%.*}.log >> ${file3}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering gowalla queries: %.2f ms\n", t/cnt);}' ${file3}
/bin/rm -f $file3
echo "gowalla ends"
 
# To run road_central
result4=road_central.log/
file4=road_central.tmp
/bin/rm -rf ${result4}
mkdir ${result4}
for query in `ls /home/data/DATASET/query4ro/*`
do
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/road_central.g ${query} ${result4}${file%.*}.txt 3 >& ${result4}${file%.*}.log
 grep "match used" ${result4}${file%.*}.log >> ${file4}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering road_central queries: %.2f ms\n", t/cnt);}' ${file4}
/bin/rm -f $file4
echo "road_central ends"

# To run watdiv100M
result5=watdiv100M.log/
file5=watdiv100M.tmp
/bin/rm -rf ${result5}
mkdir ${result5}
for query in `ls /home/data/DATASET/query4wat100/*`
do
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv100M.g ${query} ${result5}${file%.*}.txt 3 >& ${result5}${file%.*}.log
 grep "match used" ${result5}${file%.*}.log >> ${file5}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv100M queries: %.2f ms\n", t/cnt);}' ${file5}
/bin/rm -f $file5
echo "watdiv100M ends"

# To run dbpedia170M
result6=dbpedia170M.log/
file6=dbpedia170M.tmp
/bin/rm -rf ${result6}
mkdir ${result6}
for query in `ls /home/data/DATASET/query4db/*`
do
 file=${query##*/}
 ./GSI.exe /home/data/DATASET/dbpedia170M.g ${query} ${result6}${file%.*}.txt 3 >& ${result6}${file%.*}.log
 grep "match used" ${result6}${file%.*}.log >> ${file6}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering dbpedia170M queries: %.2f ms\n", t/cnt);}' ${file6}
/bin/rm -f $file6
echo "dbpedia170M ends"

