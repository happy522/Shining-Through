'''
Produces one row per .conllu file (= one episode) with the text-level feature values,
plus the columns afile / alang / akorp / astatus.

Changes to the original script (the feature calculations are unchanged):
  * no need to rename folders to "en": the conllu folders and their labels are set in CONFIG
  * works from any working directory (paths are relative to this script)
  * one output file per conllu folder + one combined file with all folders
'''
import os
import sys
import csv
from pathlib import Path

# ---------- CONFIG ----------
SCRIPT_DIR = Path(__file__).resolve().parent

# folder that contains extractors.py (and the word lists it reads)
EXTRACTORS_DIR = SCRIPT_DIR

# folder that contains your *_en_conllu folders (your "Data" folder)
DATA_DIR = SCRIPT_DIR

# conllu folder (inside DATA_DIR) -> labels written to the akorp / astatus columns (change them as you like)
CONLLU_FOLDERS = {
    "NN_France_en_conllu": {"korp": "nonnatives",   "status": "NN_France"},
    "T_France_en_conllu":  {"korp": "translations", "status": "T_France"},
}

LANGUAGE = "en"                  # all texts are English -> the English support lists are used
COMBINED_OUT = "all_out.csv"     # all folders in one table (tab-separated, like the others)
# ----------------------------

EXTRACTORS_DIR = Path(EXTRACTORS_DIR).resolve()
DATA_DIR = Path(DATA_DIR).resolve()
if not (EXTRACTORS_DIR / "extractors.py").is_file():
    sys.exit(f"extractors.py not found in {EXTRACTORS_DIR} - set EXTRACTORS_DIR in the CONFIG")
sys.path.insert(0, str(EXTRACTORS_DIR))
os.chdir(EXTRACTORS_DIR)         # extractors.py may open its word lists with relative paths
from extractors import *

# here, for each file we collect counts averaged over number of words or number of sentences
## muted features: passives interrog andor wdlength mark nn
keys = 'afile alang akorp astatus ' \
       'sentlength ppron possdet indef cconj whconj relativ pied correl copula ' \
       'attrib pasttense lexdens lexTTR mquantif mpred finites infs pverbals deverbals ' \
       'bypassives longpassives sconj addit advers caus tempseq epist but comp ' \
       'sup neg numcls simple demdets nnargs mhd mdd acl aux ' \
       'aux:pass ccomp nsubj:pass parataxis xcomp'.split()

basic_stats = {}
languages = [LANGUAGE]  # since all data is in english
adv_support = {}
mpred_support = {}
pseudo_deverbs = {}
vconverts = {}

additive = {}
adversative = {}
causal = {}
sequen = {}
epistem = {}

for l in languages:
    # import all the lists for the language and add them to a lang-dictionary
    adv_lst, mpred_lst, pseudo_deverbs_lst, vconverts_lst = support_all_lang(l)
    adv_support[l] = adv_lst
    mpred_support[l] = mpred_lst
    pseudo_deverbs[l] = pseudo_deverbs_lst
    vconverts[l] = vconverts_lst

    additive_lst, adversative_lst, causal_lst, sequen_lst, epistem_lst = dms_support_all_langs(l)
    additive[l] = additive_lst
    adversative[l] = adversative_lst
    causal[l] = causal_lst
    sequen[l] = sequen_lst
    epistem[l] = epistem_lst

    print('---', file=sys.stderr)
    print('Importing support lists for %s:' % l.upper(), file=sys.stderr)
    print('==%s adverbial qualtifiers' % len(adv_support[l]), file=sys.stderr)
    print('==%s modal predicative adjectives' % len(mpred_support[l]), file=sys.stderr)
    print('==%s stopwords for deverbal nouns' % len(pseudo_deverbs[l]), file=sys.stderr)
    print('==%s verbal nouns by conversion' % len(vconverts[l]), file=sys.stderr)
    print('', file=sys.stderr)
    print('Importing DM searchlists for %s:' % l.upper(), file=sys.stderr)
    print('==%s additive' % len(additive[l]), file=sys.stderr)
    print('==%s adversative' % len(adversative[l]), file=sys.stderr)
    print('==%s causative' % len(causal[l]), file=sys.stderr)
    print('==%s temporal/sequencial' % len(sequen[l]), file=sys.stderr)
    print('==%s DM of epistemic stance' % len(epistem[l]), file=sys.stderr)


def extract_features(filepath, korp, status):
    """All features for one .conllu file (same calculations as the original script)."""
    lang = LANGUAGE
    language = lang
    doc = Path(filepath).stem  # file name without extension, e.g. ep-04-03-10

    data = open(filepath, 'r', encoding='utf-8', errors='ignore').readlines()

    corp_id = lang + '_' + status + '_' + korp
    sents = get_trees(data)
    basic_stats[corp_id] = basic_stats.get(corp_id, 0) + len(sents)

    current = {}

    # call functions that operate at doc-level and write to dic for current file
    # get text parameters for normalization
    normBy_wc = wordcount(sents)
    normBy_sentnum = sents_num(sents, language)
    normBy_verbnum = verbs_num(sents, language)

    ## create text-level counters for each feature values
    ppron_res = 0
    possdet_res = 0
    anysome_res = 0
    cconj_res = 0
    sconj_res = 0
    advconj_res = 0
    relativ_res = 0
    pied_res = 0
    correl_res = 0
    copula_res = 0
    attrib_res = 0
    pasttense_res = 0
    lex_ty_res = 0
    lex_to_res = 0
    mpred_res = 0
    mquantif_res = 0
    finites_res = 0
    infs_res = 0
    pverbals_res = 0
    bypassives_res = 0
    longpassives_res = 0

    speakdiff_res = 0
    readerdiff_res = 0

    ## text-level counts
    deverbals_res = nominals(sents, language, pseudo_deverbs, vconverts)

    addit_res = count_dms(additive, sents, language)
    advers_res = count_dms(adversative, sents, language)
    caus_res = count_dms(causal, sents, language)
    tempseq_res = count_dms(sequen, sents, language)
    epist_res = count_dms(epistem, sents, language) + get_epistemic_stance(sents, language)

    but_res = but_counts(sents, language)

    ## counts for degrees of comparison normalized internally for num of adj+adv
    comp_res, sup_res = comparison_degrees(sents, language)

    neg_res = polarity(sents, language)

    # average number of clauses per sentence and ratio of simple sentences in text
    try:
        numcls_res, simple_res = sents_complexity(sents)
    except ZeroDivisionError:
        numcls_res, simple_res = 0, 0

    demdets_res = demdeterm(sents, language)
    nnargs_res = nouns_to_all(sents)

    ## run functions and get absolute freqs for each text
    for sent in sents:
        ppron_res += prsp(sent, language)[0]
        possdet_res += possdet(sent, language)[0]
        anysome_res += anysome(sent, language)[0]
        cconj_res += cconj(sent, language)[0]
        sconj_res += sconj(sent, language)[0]
        advconj_res += whconj(sent, language)[0]
        mhd = speakdiff(sent)
        if mhd:
            speakdiff_res += speakdiff(sent)
            readerdiff_res += readerdiff(sent)
        rel, _, ppiping, correlat = relativ(sent, language)
        relativ_res += rel
        pied_res += ppiping
        correl_res += correlat
        copula_res += copulas(sent)
        attrib_res += attrib(sent)[0]
        pasttense_res += pasttense(sent)
        ty, to = lex_ty_to(sent, language)  # counts of content types and tokens are needed elsewhere
        lex_ty_res += ty
        lex_to_res += to
        mpred_res += modpred(sent, language, mpred_support)[0]
        mquantif_res += advquantif(sent, language, adv_support)[0]
        finites_res += finites(sent, language)[0]
        infs_res += infinitives(sent, language, mpred_support)
        pverbals_res += participles(sent, language)
        bys, nobys = passives(sent, language)
        bypassives_res += bys
        longpassives_res += nobys

    # run functions that are doc(file)-level
    avsents = av_s_length(sents, language)
    current['sentlength'] = avsents

    # normalisation for the absolute sentence-level freqs
    try:
        ## by number of words in the text
        current['ppron'] = ppron_res / normBy_wc
        current['possdet'] = possdet_res / normBy_wc
        current['indef'] = anysome_res / normBy_wc
        current['cconj'] = cconj_res / normBy_sentnum
        current['sconj'] = sconj_res / normBy_sentnum
        current['whconj'] = advconj_res / normBy_sentnum
        current['lexdens'] = lex_ty_res / normBy_wc
        current['mquantif'] = mquantif_res / normBy_wc

        current['lexTTR'] = lex_ty_res / lex_to_res

        current['mpred'] = mpred_res / normBy_sentnum

        ## by number of sentences
        current['mhd'] = speakdiff_res / normBy_sentnum
        current['mdd'] = readerdiff_res / normBy_sentnum
        current['relativ'] = relativ_res / normBy_sentnum
        current['pied'] = pied_res / normBy_sentnum
        current['correl'] = correl_res / normBy_sentnum
        current['copula'] = copula_res / normBy_sentnum
        current['attrib'] = attrib_res / normBy_sentnum
        current['pasttense'] = pasttense_res / normBy_sentnum
        current['finites'] = finites_res / normBy_verbnum

        ## normalized by number of verbs
        current['infs'] = infs_res / normBy_verbnum
        current['pverbals'] = pverbals_res / normBy_verbnum
        current['deverbals'] = deverbals_res / normBy_verbnum

        current['bypassives'] = bypassives_res / normBy_sentnum
        current['longpassives'] = longpassives_res / normBy_sentnum
        ## 5 semantic groups of DMs + but
        current['addit'] = addit_res / normBy_sentnum
        current['advers'] = advers_res / normBy_sentnum
        current['caus'] = caus_res / normBy_sentnum
        current['tempseq'] = tempseq_res / normBy_sentnum
        current['epist'] = epist_res / normBy_sentnum
        current['but'] = but_res / normBy_sentnum
        current['comp'] = comp_res
        current['sup'] = sup_res
        current['neg'] = neg_res / normBy_sentnum
        current['numcls'] = numcls_res
        current['simple'] = simple_res
        current['demdets'] = demdets_res / normBy_wc
        current['nnargs'] = nnargs_res
    except ZeroDivisionError:
        for k in ['ppron', 'possdet', 'indef', 'cconj', 'sconj', 'whconj', 'lexdens', 'mquantif',
                  'lexTTR', 'mpred', 'mhd', 'mdd', 'relativ', 'pied', 'correl', 'copula', 'attrib',
                  'pasttense', 'finites', 'infs', 'pverbals', 'deverbals', 'bypassives',
                  'longpassives', 'addit', 'advers', 'caus', 'tempseq', 'epist', 'but', 'comp',
                  'sup', 'neg', 'numcls', 'simple', 'demdets', 'nnargs']:
            current[k] = 0

    ## add 7 UD features from a dict
    dep_prob_dict = ud_probabilities(sents, language)
    for k, val in dep_prob_dict.items():
        current[k] = val

    # file name, language, corpus and status
    current['afile'] = doc
    current['alang'] = lang
    current['akorp'] = korp
    current['astatus'] = status
    return current


def write_table(path, table):
    with open(path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t", lineterminator="\n")
        writer.writerow(keys)
        writer.writerows(zip(*[table[key] for key in keys]))
    print('Written %s (%s rows)' % (path, len(table[keys[0]])), file=sys.stderr)


combined = {k: [] for k in keys}

for folder_name, labels in CONLLU_FOLDERS.items():
    folder = DATA_DIR / folder_name
    if not folder.is_dir():
        print('\nSkipping %s (folder not found)' % folder, file=sys.stderr)
        continue
    files = sorted(folder.glob('*.conllu'))
    print('\nProcessing %s: %s conllu files' % (folder, len(files)), file=sys.stderr)

    master_dict = {k: [] for k in keys}
    for i, filepath in enumerate(files):
        if i % 20 == 0:
            print('I have processed %s of %s files from %s' % (i, len(files), folder_name), file=sys.stderr)
        current = extract_features(filepath, labels['korp'], labels['status'])
        for key in keys:
            master_dict[key].append(current[key])
            combined[key].append(current[key])

    if files:
        out_stem = folder_name[:-len('_en_conllu')] if folder_name.endswith('_en_conllu') else folder_name
        write_table(DATA_DIR / (out_stem + '_out.csv'), master_dict)

if combined[keys[0]]:
    write_table(DATA_DIR / COMBINED_OUT, combined)

print('\nSentences per corpus:', file=sys.stderr)
for corp_id, n in basic_stats.items():
    print('  %s: %s' % (corp_id, n), file=sys.stderr)
print('Your data is ready. Lets see whether we can see any patterns in it')