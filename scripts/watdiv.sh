#!/usr/bin/env bash

# USE the 3-th GPU

#make clean
#make

# nvprof -m gld_transactions -m gst_transactions ./GSI.exe /home/data/DATASET/gowalla/loc-gowalla_edges.g /home/data/DATASET/gowalla/query/q0.g >& gowalla.prof

# TO find out which one fails, using grep -r "match used" xxx.log
# To count using grep -r "match used" xxx.log |wc -l

data=/home/data/DATASET/

# To run watdiv10M
result=watdiv10M.log/
file=watdiv10M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat10/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv10M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv10M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv10M ends"

# To run watdiv20M
result=watdiv20M.log/
file=watdiv20M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat20/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv20M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv20M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv20M ends"

# To run watdiv30M
result=watdiv30M.log/
file=watdiv30M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat30/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv30M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv30M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv30M ends"

# To run watdiv40M
result=watdiv40M.log/
file=watdiv40M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat40/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv40M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv40M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv40M ends"

# To run watdiv50M
result=watdiv50M.log/
file=watdiv50M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat50/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv50M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv50M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv50M ends"

# To run watdiv60M
result=watdiv60M.log/
file=watdiv60M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat60/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv60M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv60M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv60M ends"

# To run watdiv70M
result=watdiv70M.log/
file=watdiv70M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat70/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv70M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv70M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv70M ends"

# To run watdiv80M
result=watdiv80M.log/
file=watdiv80M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat80/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv80M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv80M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv80M ends"

# To run watdiv90M
result=watdiv90M.log/
file=watdiv90M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat90/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv90M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv90M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv90M ends"


# To run watdiv100M
result=watdiv100M.log/
file=watdiv100M.tmp
/bin/rm -rf ${result}
mkdir ${result}
for query in `ls /home/data/DATASET/query4wat100/*`
do
 qq=${query##*/}
 ./GSI.exe /home/data/DATASET/watdiv100M.g ${query} ${result}${qq%.*}.txt 3 >& ${result}${qq%.*}.log
 grep "match used" ${result}${qq%.*}.log >> ${file}
done
awk 'BEGIN{t=0.0;cnt=0}{t = t + $3;cnt=cnt+1}END{printf("the average time of answering watdiv100M queries: %.2f ms\n", t/cnt);}' ${file}
#/bin/rm -f $file
echo "watdiv100M ends"

