#!/bin/bash
#requires following fastq file format: AMPLICON_sample-xxx.fastq.gz
#requires AR's standard amplicon naming convention
#organize into folders by amplicon
folders=(H2 H3 H4 R2 E1 FF FF1 SaH3)
for folder in ${folders[*]}; do
mkdir -p $folder
for file in $( ls ${folder}-* ); do
mv $file ${folder}/${file}
done
done
for amplicon in $( ls ); do
cd $amplicon
#create batch settings txt file
echo r1'\t'n >> ${amplicon}.txt
#lookup sgRNA and amplicon sequence from standard files
g=`cat /guides+amplicons/${amplicon}_sgRNA.txt`
a=`cat /guides+amplicons/${amplicon}_amplicon.txt`
#add r1's and names to batch settings txt file
for sample in $( ls *.gz ); do
echo ${sample}'\t'${sample:0:$(expr ${#sample} - 9)} >> ${amplicon}.txt
done
#run crispressobatch
docker run -v ${PWD}:/DATA -w /DATA -i pinellolab/crispresso2 CRISPRessoBatch --batch_settings
${amplicon}.txt -g $g -a $a -w 30 –wc -10 -q 30 -p 12
cd ..
done