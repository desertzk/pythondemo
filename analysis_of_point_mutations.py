import pandas as pd
import Bio
from Bio import SeqIO
import glob
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import glob
import os
from Bio.SeqRecord import SeqRecord
files = [x[5:] for x in glob.glob('data/*.csv')]
chromosomes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,'X','Y']
os.mkdir('data+flanks')
for file in files:
data = pd.read_csv('data/' + file)
flank_seqs = {}
for chromosome in chromosomes:
print file, chromosome
dchr = data.loc[data['Chrom'] == 'chr' + str(chromosome)]
record = SeqIO.read('mm10/chr' + str(chromosome) + '.fa', 'fasta')
for index in dchr.index:
flank_seqs[index] = str(record.seq[int(dchr.loc[index]['Pos'])-1-
20:int(dchr.loc[index]['Pos'])-1+21])
data['Flank_seqs'] = pd.Series(flank_seqs)
data.to_csv('data+flanks/' + file)
#reverse complement flanks for all G>something SNPs
#remove all nonC>T or G>A SNPs
files = [x[12:] for x in glob.glob('data+flanks/*.csv')]
os.mkdir('data+flanks_C20_5p3p')
for file in files:
data = pd.read_csv('data+flanks/' + file)
data.dropna(subset=['Mutation'], inplace=True)
flank_seqs = {int(index) : data.loc[index]['Flank_seqs'] for index in data.index}
#find all G>something SNPs
gdata = data[data['Mutation'].str.contains('G>')]
#reverse complement flanks for all G>something SNPs
for index in gdata.index:
flank_seqs[int(index)] = str(Seq(str(flank_seqs[index]),
IUPAC.unambiguous_dna).reverse_complement())
data['Flank_seqs_C20_5p3p'] = pd.Series(flank_seqs)
data = data[data['Mutation'].str.contains('C>T|G>A')]
data.to_csv('data+flanks_C20_5p3p/' + file)
files = [x[21:] for x in glob.glob('data+flanks_C20_5p3p/*.csv')]
df = pd.DataFrame()
28
for file in files:
df_new = pd.read_csv('data+flanks_C20_5p3p/' + file)
df = df.append(df_new)
df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
df.reset_index(inplace=True)
df.drop(columns = ['index'], inplace=True)
sequences = []
for index in df.index:
record = SeqRecord(Seq(df.loc[index]['Flank_seqs_C20_5p3p'], IUPAC.unambiguous_dna),
id=str(index))
sequences.append(record)
SeqIO.write(sequences, 'data.fasta', 'fasta')