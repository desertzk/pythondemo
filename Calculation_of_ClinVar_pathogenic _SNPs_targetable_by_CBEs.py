import numpy as np
import pandas as pd
import regex
import re
from Bio import SeqIO
import Bio
import os
###CHANGE PAM AND WINDOW INFO HERE###
PAMs = ['NGG']
windows = {'NGG':(4,14)}
###CHANGE PAM AND WINDOW INFO HERE###
def RC(seq):
    encoder = {'A':'T','T':'A','C':'G','G':'C','N':'N','R':'Y','Y':'R', 'M':'K', 'K':'M', 'S':'S', 'W':'W', 'H':'D', 'B':'V', 'V':'B',
    'D':'H'}
    rc = ''
    for n in reversed(seq):
        rc += encoder[n]
        return rc


def create_PAM(pam):
    encoder ={'A':'A','T':'T','G':'G','C':'C','R':'[A|G]','Y':'[C|T]','N':'[A|T|C|G]','M':'[A|C]','K':'[G|T]','S':'[C|G]','W':'[A|T]','H':'[A|C|T]','B':'[C|G|T]','V':'[A|C|G]','D':'[A|G|T]'}
    enc_pam = {'f':'','r':''}
    rc_pam = RC(pam)
    for n,m in zip(pam, rc_pam):
        enc_pam['f'] += encoder[n]
        enc_pam['r'] += encoder[m]
    return enc_pam


ClinVar=pd.read_csv('2019-05-06-variant_summary.csv', encoding = "ISO-8859-1")
Phenotypes=pd.read_csv('DiseaseNames.csv')
PhenotypeDict=dict(zip(Phenotypes.CUI, Phenotypes.name))
#open flanking sequence fasta files for all Y-type pathogenic human SNPs (includes both C>T and T>Cref>variant)
#downloaded as fasta file from:
#http://www.ncbi.nlm.nih.gov/snp/?term=((%22pathogenic%22%5BClinical+Significance%5D+AND+%22snp%22%5BSNP+Class%5D+AND+homo+sapiens%5BOrganism%5D+)+AND(%22y%22%5BAllele%5D))
handle = open("YFasta.txt", "rU")
flanks={}
#save as a dictionary keyed on rsID as an Integer with values being 25nt of flanking sequence on each side of the SNP
for record in SeqIO.parse(handle, "fasta") :
    flanks[int(record.id.split("|")[-1].strip('rs'))]=regex.findall('.{25}[^A,T,C,G].{25}', record.seq.tostring())
    handle.close()

# clinvar may refer to the opposite strand that was used in dbSNP;
# we want to allow clinvar reference alleles A and T with alternate alleles G and C respectively
# we do not want to allow reference alleles G and C with alternate alleles A and T respectively; these Y-type SNPs must be removed
ClinVar_mod=ClinVar[(ClinVar.ReferenceAllele=='A') |
(ClinVar.ReferenceAllele=='T')].drop_duplicates('#AlleleID')
#merge flanking sequences to the CtoT frame on rsID
F=pd.DataFrame({'RS# (dbSNP)': list(flanks.keys()), 'Flanks': [x for x in flanks.values()]})
CtoT=F.merge(ClinVar_mod, left_on='RS# (dbSNP)', right_on='RS# (dbSNP)', how='inner')
#open flanking sequence fasta files for all R-type pathogenic human SNPs (includes both G>A and A>Gref>variant)
#downloaded as fasta file from:
#http://www.ncbi.nlm.nih.gov/snp/?term=((%22pathogenic%22%5BClinical+Significance%5D+AND+%22snp%22%5BSNP+Class%5D+AND+homo+sapiens%5BOrganism%5D+)+AND(%22r%22%5BAllele%5D))
handle = open("RFasta.txt", "rU")
flanks={}
#save as a dictionary keyed on rsID as an Integer with values being 25nt of flanking sequence on each side of
the SNP
for record in SeqIO.parse(handle, "fasta") :
flanks[int(record.id.split("|")[-1].strip('rs'))]=regex.findall('.{25}[^A,T,C,G].{25}', record.seq.tostring())
handle.close()
#merge flanking sequences to the CtoT frame on rsID
F=pd.DataFrame({'RS# (dbSNP)': list(flanks.keys()), 'Flanks': [x for x in flanks.values()]})
GtoA=F.merge(ClinVar_mod, left_on='RS# (dbSNP)', right_on='RS# (dbSNP)', how='inner')
#empty lists to later combine data for all PAMs
hasPAM_CtoT_dfs =[]
hasPAM_GtoA_dfs =[]
singleC_CtoT_dfs = []
singleC_GtoA_dfs = []
os.chdir('output')
for PAM in PAMs:
#define window limits and the length of the pam including all N residues
enc_pam = create_PAM(PAM)
windowstart = windows[PAM][0]
windowend = windows[PAM][1]
windowlen=windowend-windowstart+1
lenpam=len(PAM)
CtoTmod = CtoT
CtoTmod['gRNAs']=None
CtoTmod['gRNAall']=None
for i in range(len(CtoTmod)):
print i
if type(CtoTmod.iloc[i].Flanks)==list and CtoTmod.iloc[i].Flanks!=[]:
test=CtoTmod.iloc[i].Flanks[0]
# define a potential gRNA spacer for each window positioning
gRNAoptions=[test[(26-windowstart-j):(26-windowstart-j+lenpam+20)] for j in
range(windowlen)]
#if there is an appropriate PAM placed for a given gRNA spacer
#save tuple of gRNA spacer, and the position of off-target Cs in the window
gRNA=[(gRNAoptions[k],[x.start()+1 for x in re.finditer('C',gRNAoptions[k]) if
windowstart-1<x.start()+1<windowend+1]) for k in range(len(gRNAoptions)) if regex.match(enc_pam['f'],
gRNAoptions[k][-lenpam:])]
33
gRNAsingleC=[]
for g,c in gRNA:
#if the target C is the only C in the window save this as a single C site
if g[windowstart-1:windowend].count('C')==0:
gRNAsingleC.append(g)
#OPTIONAL uncomment the ELIF statement if you are interest in filtered based
upon position of off-target C
#if the target C is expected to be editted more efficiently than the off-target Cs,
also save as a single C Site
#elif all([p<priority[x] for x in c]):
#gRNAsingleC.append(g)
CtoTmod.gRNAs.iloc[i]=gRNAsingleC
CtoTmod.gRNAall.iloc[i]=[g for g,c in gRNA]
GtoAmod = GtoA
GtoAmod['gRNAs']=None
GtoAmod['gRNAall']=None
for i in range(len(GtoAmod)):
print i
if type(GtoAmod.iloc[i].Flanks)==list and GtoAmod.iloc[i].Flanks!=[]:
test=GtoAmod.iloc[i].Flanks[0]
gRNAoptions=[test[(25+windowstart+j-20-lenpam):(25+windowstart+j)] for j in
range(windowlen)]
gRNA=[(gRNAoptions[k],[20+lenpam-x.start() for x in re.finditer('G',gRNAoptions[k]) if
windowstart-1<20+lenpam-x.start()<windowend+1]) for k in range(len(gRNAoptions)) if
regex.match(enc_pam['r'], gRNAoptions[k][:lenpam])]
gRNAsingleC=[]
for g,c in gRNA:
if g[20+lenpam-windowstart-windowlen+1:20+lenpam�windowstart+1].count('G')==0:
gRNAsingleC.append(g)
#elif all([p<priority[x] for x in c]):
#gRNAsingleC.append(g)
GtoAmod.gRNAs.iloc[i]=gRNAsingleC
GtoAmod.gRNAall.iloc[i]=[g for g,c in gRNA]
#merge in phenotypes based upon MedGen IDs; remove redundant columns
CtoTmod=CtoTmod[['RS# (dbSNP)','GeneSymbol','Name', 'PhenotypeIDs', 'Origin', 'ReviewStatus',
'NumberSubmitters', 'LastEvaluated', 'gRNAs', 'gRNAall']]
ids=[re.findall('MedGen:C.{7}', x) for x in CtoTmod.PhenotypeIDs.values]
CtoTmod['Phenotypes']=[[PhenotypeDict[y.lstrip('MedGen:')] for y in x if y.lstrip('MedGen:') in
PhenotypeDict.keys()] for x in ids]
CtoTmod.drop('PhenotypeIDs', inplace=True, axis=1)
GtoAmod=GtoAmod[['RS# (dbSNP)','GeneSymbol','Name', 'PhenotypeIDs', 'Origin', 'ReviewStatus',
'NumberSubmitters', 'LastEvaluated', 'gRNAs', 'gRNAall']]
ids=[re.findall('MedGen:C.{7}', x) for x in GtoAmod.PhenotypeIDs.values]
GtoAmod['Phenotypes']=[[PhenotypeDict[y.lstrip('MedGen:')] for y in x if y.lstrip('MedGen:') in
PhenotypeDict.keys()] for x in ids]
GtoAmod.drop('PhenotypeIDs', inplace=True, axis=1)
CtoTmod.to_csv('pathogenic_CtoT_all.csv')
GtoAmod.to_csv('pathogenic_GtoA_all.csv')
pathogenic_CtoT_hasPAM=CtoTmod[[type(x)==list and x!=[] for x in CtoTmod.gRNAall]]
pathogenic_GtoA_hasPAM=GtoAmod[[type(x)==list and x!=[] for x in GtoAmod.gRNAall]]
34
pathogenic_GtoA_hasPAM.to_csv('pathogenic_GtoA_has_'+PAM+'_PAM.csv')
pathogenic_CtoT_hasPAM.to_csv('pathogenic_CtoT_has_'+PAM+'_PAM.csv')
hasPAM_CtoT_dfs.append(pathogenic_CtoT_hasPAM)
hasPAM_GtoA_dfs.append(pathogenic_GtoA_hasPAM)
pathogenic_CtoT_SingleC=CtoTmod[[type(x)==list and x!=[] for x in CtoTmod.gRNAs]]
pathogenic_GtoA_SingleC=GtoAmod[[type(x)==list and x!=[] for x in GtoAmod.gRNAs]]
pathogenic_GtoA_SingleC.to_csv('pathogenic_GtoA_'+PAM+'_PAM_SingleC.csv')
pathogenic_CtoT_SingleC.to_csv('pathogenic_CtoT_'+PAM+'_PAM_SingleC.csv')
singleC_CtoT_dfs.append(pathogenic_CtoT_SingleC)
singleC_GtoA_dfs.append(pathogenic_GtoA_SingleC)
with open('Summary_'+PAM+'.txt', "w") as text_file:
text_file.write("singleC %s \n" %
(len(pathogenic_CtoT_SingleC)+len(pathogenic_GtoA_SingleC)))
text_file.write("hasPAM %s \n" %
(len(pathogenic_CtoT_hasPAM)+len(pathogenic_GtoA_hasPAM)))
text_file.write("Pathogenic SNPs that can be targeted with BE %s" %
(len(CtoTmod)+len(GtoAmod)))
hasPAM_CtoT_allPAMs = pd.concat(hasPAM_CtoT_dfs)
hasPAM_GtoA_allPAMs = pd.concat(hasPAM_GtoA_dfs)
singleC_CtoT_allPAMs = pd.concat(singleC_CtoT_dfs)
singleC_GtoA_allPAMs = pd.concat(singleC_GtoA_dfs)
#remove duplicates
hasPAM_CtoT_allPAMs = hasPAM_CtoT_allPAMs[~hasPAM_CtoT_allPAMs.index.duplicated(keep='first')]
hasPAM_GtoA_allPAMs = hasPAM_GtoA_allPAMs[~hasPAM_GtoA_allPAMs.index.duplicated(keep='first')]
singleC_CtoT_allPAMs = singleC_CtoT_allPAMs[~singleC_CtoT_allPAMs.index.duplicated(keep='first')]
singleC_GtoA_allPAMs = singleC_GtoA_allPAMs[~singleC_GtoA_allPAMs.index.duplicated(keep='first')]
with open('Summary_allPAMs.txt', "w") as text_file:
text_file.write("singleC %s \n" % (len(singleC_CtoT_allPAMs)+len(singleC_GtoA_allPAMs)))
text_file.write("hasPAM %s \n" % (len(hasPAM_CtoT_allPAMs)+len(hasPAM_GtoA_allPAMs)))
text_file.write("Pathogenic SNPs that can be targeted with BE %s" % (len(CtoTmod)+len(GtoAmod)))