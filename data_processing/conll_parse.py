#!/usr/bin/env python
# Code authored by Prakhar Gupta and adapted from https://github.com/nlplab/brat/blob/master/tools/conll09tostandoff.py
# Convert CoNLL 2009 format file into brat-flavored standoff and a
# reconstruction of the original text.



import codecs
import os
import sys
import csv
import argparse
import pprint
pp = pprint.PrettyPrinter()
# maximum number of sentences to include in single output document
# (if None, doesn't split into documents)
MAX_DOC_SENTENCES = None

# whether to output an explicit root note
OUTPUT_ROOT = True
# the string to use to represent the root node
ROOT_STR = 'ROOT'
ROOT_POS = 'ROOT'
ROOT_FEAT = ''

INPUT_ENCODING = "UTF-8"
OUTPUT_ENCODING = "UTF-8"

# fields of interest in input data; full list: ID FORM LEMMA PLEMMA
# POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs
# (http://ufal.mff.cuni.cz/conll2009-st/task-description.html)

F_ID, F_FORM, F_LEMMA, F_POS, F_FEAT, F_HEAD, F_DEPREL, F_FILLPRED, F_PRED, F_APRED1 = list(range(
    10))

output_directory = None

# rewrites for characters appearing in CoNLL-X types that cannot be
# directly used in identifiers in brat-flavored standoff
charmap = {
    '<': '_lt_',
    '>': '_gt_',
    '+': '_plus_',
    '?': '_question_',
    '&': '_amp_',
    ':': '_colon_',
    '.': '_period_',
    '!': '_exclamation_',
}


def maptype(s):
    return "".join([charmap.get(c, c) for c in s])


def tokstr(start, end, ttype, idnum, text):
    # sanity checks
    assert '\n' not in text, "ERROR: newline in entity '%s'" % (text)
    assert text == text.strip(), "ERROR: tagged span contains extra whitespace: '%s'" % (text)
    return "T%d\t%s %d %d\t%s" % (idnum, maptype(ttype), start, end, text)


def featstr(lemma, feats, idnum):
    return "#%d\tData T%d\tLemma: %s, Feats: %s" % (idnum, idnum, lemma, feats)


def depstr(depid, headid, rel, idnum):
    return "R%d\t%s Arg1:T%d Arg2:T%d" % (idnum, maptype(rel), headid, depid)


def output(infn, docnum, sentences):
    global output_directory

    if output_directory is None:
        txtout = codecs.getwriter(OUTPUT_ENCODING)(sys.stdout)
        soout = codecs.getwriter(OUTPUT_ENCODING)(sys.stdout)
    else:
        # add doc numbering if there is a sentence count limit,
        # implying multiple outputs per input
        if MAX_DOC_SENTENCES:
            outfnbase = os.path.basename(infn) + '-doc-' + str(docnum)
        else:
            outfnbase = os.path.basename(infn)
        outfn = os.path.join(output_directory, outfnbase)
        txtout = codecs.open(outfn + '.txt', 'wt')#, encoding=OUTPUT_ENCODING)
        soout = codecs.open(outfn + '.ann', 'wt')#, encoding=OUTPUT_ENCODING)

    offset, idnum, ridnum = 0, 1, 1

    doctext = ""

    for si, sentence in enumerate(sentences):
        tokens, deps = sentence

        # store mapping from per-sentence token sequence IDs to
        # document-unique token IDs
        idmap = {}

        # output tokens
        prev_form = None

        if OUTPUT_ROOT:
            # add an explicit root node with seq ID 0 (zero)
            tokens[0] = (ROOT_STR, ROOT_STR, ROOT_POS, ROOT_FEAT, ROOT_FEAT)

        for id_ in tokens:
            # print(tokens[id_])
            form, lemma, pos, feat, apred = tokens[id_]

            if prev_form is not None:
                doctext = doctext + ' '
                offset += 1

            # output a token annotation
            # print(tokstr(
            #     offset, offset + len(form), pos, idnum, form), file=soout)
            # print(featstr(lemma, feat, idnum), file=soout)

            print >>soout, (tokstr(
                offset, offset + len(form), pos, idnum, form))
            print >>soout, (featstr(lemma, feat, idnum))
            assert id_ not in idmap, "Error in data: dup ID"
            idmap[id_] = idnum
            idnum += 1

            doctext = doctext + form
            offset += len(form)

            prev_form = form

        # output dependencies
        for head in deps:
            for dep in deps[head]:
                for rel in deps[head][dep]:
                    # if root is not added, skip deps to the root (idx 0)
                    if not OUTPUT_ROOT and head == 0:
                        continue

                    # print(depstr(
                    #     idmap[dep], idmap[head], rel, ridnum), file=soout)

                    print >>soout,(depstr(
                        idmap[dep], idmap[head], rel, ridnum))
                    ridnum += 1

        if si + 1 != len(sentences):
            doctext = doctext + '\n'
            offset += 1

    # print(doctext, file=txtout)
    print >>txtout, (doctext)

def read_sentences_org(fn):
    """Read sentences in CoNLL format.

    Return list of sentences, each represented as list of fields.
    """
    # original author: @fginter
    all_sentences = []
    sentences = [[]]
    with codecs.open(fn, 'rU', INPUT_ENCODING) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                # if len(sentences)>=2 and len(sentences[-1]) == len(sentences[-2]):
                #     same = True
                #     for i,j in zip(sentences[-1], sentences[-2]):
                #         # 1 is the token text
                #         if i[1] != j[1]:
                #             same = False
                #             break
                #     if same:
                #         sentences.pop()

                continue
            # igore lines starting with "#" as comments
            if line and line[0] == "#":
                continue
            cols = line.split('\t')
            # break sentences on token index instead of blank line;
            # the latter isn't reliably written by all generators
            if cols[0] == '1' and sentences[-1]:
                sentences.append([])
            sentences[-1].append(cols)

    print(len(sentences))
    return sentences


def read_sentences(fn):
    """Read sentences in CoNLL format.

    Return list of sentences, each represented as list of fields.
    """
    # original author: @fginter
    all_sentences = []
    sentences = []
    count = 0
    with codecs.open(fn, 'rU', INPUT_ENCODING) as f:
        for line in f:
            count+=1
            line = line.rstrip()
            if not line:
                # if len(sentences)>=2 and len(sentences[-1]) == len(sentences[-2]):
                #     same = True
                #     for i,j in zip(sentences[-1], sentences[-2]):
                #         # 1 is the token text
                #         if i[1] != j[1]:
                #             same = False
                #             break
                #     if same:
                #         sentences.pop()
                same = False
                if len(all_sentences)>=1 and len(sentences) == len(all_sentences[-1][-1]):
                    inner_same = True
                    for i,j in zip(sentences, all_sentences[-1][-1]):
                        # 1 is the token text
                        if i[1] != j[1]:
                            inner_same = False
                            break
                    if inner_same:
                        same = True
                        all_sentences[-1].append(sentences)
                if not same:
                    all_sentences.append([sentences])

                # print('len last', len(all_sentences[-1]))
                continue
            # igore lines starting with "#" as comments
            if line and line[0] == "#":
                continue
            cols = line.split('\t')
            # break sentences on token index instead of blank line;
            # the latter isn't reliably written by all generators
            if cols[0] == '1':
                sentences = []
            sentences.append(cols)

    print(len(all_sentences), 'len(all_sentences)')
    return all_sentences


def read_sentences_iterator(fn):
    """Read sentences in CoNLL format.

    Return list of sentences, each represented as list of fields.
    """
    # original author: @fginter
    all_sentences = []
    sentences = []
    count = 0
    prev_sentence_id = -1
    with codecs.open(fn, 'rU', INPUT_ENCODING) as f:
        for line in f:
            count+=1
            line = line.rstrip()
            if not line:
                # all_sentences [group][sentence][column_line][column]
                all_sentences.append(sentences)
                continue
            # igore lines starting with "#" as comments
            if line and line[0] == "#":
                continue
            cols = line.split('\t')
            # break sentences on token index instead of blank line;
            # the latter isn't reliably written by all generators
            if cols[0] == '1':
                # import pdb; pdb.set_trace()

                if len(all_sentences)>=1 and all_sentences[-1][-1][6] != cols[6]:
                    prev_sentence_id = int(all_sentences[-1][-1][6])
                    yield prev_sentence_id, all_sentences
                    all_sentences = []

                sentences = []

            sentences.append(cols)

        #after finish
        if len(all_sentences)>0:
            prev_sentence_id = int(all_sentences[-1][-1][6])
            yield prev_sentence_id, all_sentences
    # print(len(all_sentences), 'len(all_sentences)')
    # return all_sentences



def resolve_format(sentences, options):
    fields = {}

    # TODO: identify CoNLL format variant by heuristics on the sentences

    # CoNLL'09 field structure, using gold instead of predicted (e.g.
    # POS instead of PPOS).
    fields[F_ID] = 0
    fields[F_FORM] = 1
    fields[F_LEMMA] = 2
    # PLEMMA = 3
    fields[F_POS] = 4
    # PPOS = 5
    fields[F_FEAT] = 6
    # PFEAT = 7
    fields[F_HEAD] = 8
    # PHEAD = 9
    fields[F_DEPREL] = 10
    # PDEPREL = 11
    fields[F_FILLPRED] = 12
    fields[F_PRED] = 13
    fields[F_APRED1] = 14

    return fields

def get_format(options):
    fields = {}

    # TODO: identify CoNLL format variant by heuristics on the sentences

    # CoNLL'09 field structure, using gold instead of predicted (e.g.
    # POS instead of PPOS).
    fields[F_ID] = 0
    fields[F_FORM] = 1
    fields[F_LEMMA] = 2
    # PLEMMA = 3
    fields[F_POS] = 4
    # PPOS = 5
    fields[F_FEAT] = 6
    # PFEAT = 7
    fields[F_HEAD] = 8
    # PHEAD = 9
    fields[F_DEPREL] = 10
    # PDEPREL = 11
    fields[F_FILLPRED] = 12
    fields[F_PRED] = 13
    fields[F_APRED1] = 14

    return fields


def mark_dependencies(dependency, head, dependent, deprel):
    if head not in dependency:
        dependency[head] = {}
    if dependent not in dependency[head]:
        dependency[head][dependent] = []
    dependency[head][dependent].append(deprel)
    return dependency


def process_sentence(sentence, fieldmap):
    # dependencies represented as dict of dicts of lists of dep types
    # dependency[head][dependent] = [type1, type2, ...]
    dependency = {}
    # tokens represented as dict indexed by ID, values (form, lemma,
    # POS, feat)
    token = {}

    # import pdb;pdb.set_trace()
    for fields in sentence:
        id_ = int(fields[fieldmap[F_ID]])
        form = fields[fieldmap[F_FORM]]
        lemma = fields[fieldmap[F_LEMMA]]
        pos = fields[fieldmap[F_POS]]
        feat = fields[fieldmap[F_FEAT]]
        try:
            head = int(fields[fieldmap[F_HEAD]])
        except ValueError:
            # assert fields[fieldmap[F_HEAD]] == 'ROOT', \
            #     'error: unexpected head: %s' % fields[fieldmap[F_HEAD]]
            head = 0
        deprel = fields[fieldmap[F_DEPREL]]
        fillpred = fields[fieldmap[F_FILLPRED]]
        pred = fields[fieldmap[F_PRED]]
        apreds = fields[fieldmap[F_APRED1]:]

        # mark_dependencies(dependency, head, id_, deprel)
        assert id_ not in token
        token[id_] = (form, lemma, fillpred, pred, apreds)

    return token, dependency


def get_frames_sentence_group(sentences, fieldmap):
    # dependencies represented as dict of dicts of lists of dep types
    # dependency[head][dependent] = [type1, type2, ...]
    dependency = {}
    # tokens represented as dict indexed by ID, values (form, lemma,
    # POS, feat)
    token = {}
    sentence_text = {}

    # import pdb;pdb.set_trace()
    for i, sentence in enumerate(sentences):

        for fields in sentence:
            id_ = int(fields[fieldmap[F_ID]])
            form = fields[fieldmap[F_FORM]]

            lemma = fields[fieldmap[F_LEMMA]]
            pos = fields[fieldmap[F_POS]]
            feat = fields[fieldmap[F_FEAT]]
            try:
                head = int(fields[fieldmap[F_HEAD]])
            except ValueError:
                # assert fields[fieldmap[F_HEAD]] == 'ROOT', \
                #     'error: unexpected head: %s' % fields[fieldmap[F_HEAD]]
                head = 0
            deprel = fields[fieldmap[F_DEPREL]]
            fillpred = fields[fieldmap[F_FILLPRED]]
            pred = fields[fieldmap[F_PRED]]
            apreds = fields[fieldmap[F_APRED1]:]

            # mark_dependencies(dependency, head, id_, deprel)
            # assert id_ not in token
            if id_ not in token or token[id_] == '_':
                token[id_] = (pred)
            if i == len(sentences)-1:
                sentence_text[id_] = (form)

    return token, sentence_text

def get_plain_text_frames(frames_map, sentence_map):
    frames_list = []
    token_list = []
    len_sentence = max(sentence_map, key=int)
    for i in range(len_sentence):
        token_list.append(sentence_map[i+1])
        frames_list.append(frames_map[i+1])

    return " ".join(frames_list), " ".join(token_list)



def process_org(fn, options=None):
    docnum = 1
    sentences = read_sentences_org(fn)
    # import pdb; pdb.set_trace()
    fieldmap = resolve_format(sentences, options)
    processed = []

    for i, sentence in enumerate(sentences):
        token, dependency = process_sentence(sentence, fieldmap)
        processed.append((token, dependency))

        # limit sentences per output "document"
        if MAX_DOC_SENTENCES and len(processed) >= MAX_DOC_SENTENCES:
            output(fn, docnum, processed)
            processed = []
            docnum += 1

def process(fn, options=None):

    global output_directory
    outfnbase = os.path.basename(fn)
    outfn = os.path.join(output_directory, outfnbase)

    #clear the file
    csv_file = open(outfn + '.csv', "w")
    csv_file.close()


    csv_file = open(outfn + '.csv', "a")
    writer = csv.writer(csv_file)

    docnum = 0
    fieldmap = get_format(options)
    chunk_size = 1000
    # total_sentences = 2317679
    for sentence_id, sentences in read_sentences_iterator(fn):
        # pprint.pprint(sentences, indent=4)
        # print(len(sentences[0]))
        if docnum%chunk_size ==0:
            print(docnum, " processed")

        while sentence_id>docnum:
            docnum+=1
            writer.writerow(["", ""])

        # processed = []
        frames_map, sentence_map = get_frames_sentence_group(sentences, fieldmap)
        # processed.append((frames_map, sentence_map))
        frames_text, sentence_text = get_plain_text_frames(frames_map, sentence_map)
        # import pdb; pdb.set_trace()
        # print(frames_text)
        writer.writerow([sentence_text, frames_text])

        docnum += 1
        # sys.stderr.flush()
    # while total_sentences>docnum:
    #     docnum+=1
    #     writer.writerow(["", ""])
    print(docnum, " processed")
    csv_file.flush()
    csv_file.close()



def main(argv):

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--out", help="out file", default = "indi_columns/")
    # parser.add_argument("--out_dir", help="out file", default = "sesame/outs")


    # opt = parser.parse_args()

    global output_directory

    # Take an optional "-o" arg specifying an output directory for the results
    output_directory = None
    filenames = argv[1:]
    if len(argv) > 2 and argv[1] == "-o":
        output_directory = argv[2]
        print(sys.stderr, ("Writing output to %s" % output_directory))
        filenames = argv[3:]

    fail_count = 0
    for fn in filenames:
        try:
            # process_org(fn)
            process(fn)
        except Exception as e:
            str(e).encode(OUTPUT_ENCODING)
            raise
            #print >> sys.stderr, "Error processing %s: %s" % (fn, m)
            #fail_count += 1

    if fail_count > 0:
        print("""
##############################################################################
#
# WARNING: error in processing %d/%d files, output is incomplete!
#
##############################################################################
""" % (fail_count, len(filenames)))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
