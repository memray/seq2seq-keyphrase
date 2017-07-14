import os

import numpy
import shutil

import keyphrase.config
from emolga.dataset.build_dataset import deserialize_from_file
from keyphrase.dataset.keyphrase_test_dataset import load_additional_testing_data

def export_UTD():
    # prepare logging.
    config  = keyphrase.config.setup_keyphrase_all()   # load settings.

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config)

    for dataset_name, dataset in test_sets.items():
        print('Exporting %s' % str(dataset_name))

        # keep the first 400 in krapivin
        if dataset_name == 'krapivin':
            dataset['tagged_source'] = dataset['tagged_source'][:400]

        for i, d in enumerate(zip(dataset['tagged_source'], dataset['target_str'])):
            source_postag, target = d
            print('[%d/%d]' % (i, len(dataset['tagged_source'])))

            output_text = ' '.join([sp[0]+'_'+sp[1] for sp in source_postag])

            output_dir = config['baseline_data_path'] + dataset_name + '/text/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+'/'+str(i)+'.txt', 'w') as f:
                f.write(output_text)

            output_text = '\n'.join([' '.join(t) for t in target])
            tag_output_dir = config['baseline_data_path'] + dataset_name + '/keyphrase/'
            if not os.path.exists(tag_output_dir):
                os.makedirs(tag_output_dir)
            with open(tag_output_dir+'/'+str(i)+'.txt', 'w') as f:
                f.write(output_text)

class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.text       = ''
        self.phrases    = []

    def __str__(self):
        return '%s\n\t%s\n\t%s' % (self.title, self.text, str(self.phrases))

def load_text(doclist, textdir):
    for filename in os.listdir(textdir):
        with open(textdir+filename) as textfile:
            doc = Document()
            doc.name = filename[:filename.find('.txt')]

            import string
            printable = set(string.printable)

            # print((filename))
            try:
                lines = textfile.readlines()

                lines = [filter(lambda x: x in printable, l) for l in lines]

                title = lines[0].encode('ascii', 'ignore').decode('ascii', 'ignore')
                # the 2nd line is abstract title
                text  = (' '.join(lines[2:])).encode('ascii', 'ignore').decode('ascii', 'ignore')

                # if lines[1].strip().lower() != 'abstract':
                #     print('Wrong title detected : %s' % (filename))

                doc.title = title
                doc.text  = text
                doclist.append(doc)

            except UnicodeDecodeError:
                print('UnicodeDecodeError detected! %s' % filename )
    return doclist

def load_keyphrase(doclist, keyphrasedir):
    for doc in doclist:
        phrase_set = set()
        if os.path.exists(keyphrasedir + doc.name + '.keyphrases'):
            with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
        # else:
        #     print(self.keyphrasedir + doc.name + '.keyphrases Not Found')

        if os.path.exists(keyphrasedir + doc.name + '.keywords'):
            with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
        # else:
        #     print(self.keyphrasedir + doc.name + '.keywords Not Found')

        doc.phrases = list(phrase_set)
    return doclist

def get_doc(text_dir, phrase_dir):
    '''
    :return: a list of dict instead of the Document object
    '''
    doclist = []
    doclist = load_text(doclist, text_dir)
    doclist = load_keyphrase(doclist, phrase_dir)

    for d in doclist:
        print(d)

    return doclist

def export_maui():
    # prepare logging.
    config  = keyphrase.config.setup_keyphrase_all()   # load settings.

    data_infos = [['inspec_train',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/INSPEC/train_validation_texts/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/INSPEC/gold_standard_train_validation/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/inspec/train/'],
                  ['inspec_test',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/INSPEC/test_texts/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/INSPEC/gold_standard_test/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/inspec/test/'],
                  ['nus',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/NUS/abstract_introduction_texts/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/NUS/gold_standard_keyphrases/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/nus/'],
                  ['semeval_train',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/SemEval/train+trial/all_texts/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/SemEval/train+trial/gold_standard_keyphrases_3/',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/semeval/train/'],
                  ['semeval_test',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/testing-data/SemEval/test/',
                   '',
                   '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/semeval/test/']
                  ]

    for dataset_name, text_dir, target_dir, output_dir in data_infos:
        print('Exporting %s in %s' % (str(dataset_name), text_dir))
        file_names = [file_name[: file_name.index('.txt')] for file_name in os.listdir(text_dir)]

        if not os.path.exists(text_dir):
            os.makedirs(text_dir)
        if target_dir!="" and not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, file_name in enumerate(file_names):
            print('Exporting %d. %s - %s' % (i, dataset_name, file_name))
            print('Text file %s' % (text_dir + str(i) + '.txt'))
            with open(text_dir + file_name + '.txt', 'r') as inf:
                text = inf.read()
                with open(output_dir + str(i) + '.txt', 'w') as outf:
                    outf.write(text)

            print('Target file %s' % (target_dir + file_name + '.keyphrases'))
            targets = []
            if target_dir.strip() != '':
                with open(target_dir + file_name + '.keyphrases', 'r') as inf:
                    targets.extend([l.strip() for l in inf.readlines()])
                with open(target_dir + file_name + '.keywords', 'r') as inf:
                    targets.extend([l.strip() for l in inf.readlines()])
                with open(output_dir + str(i) + '.key', 'w') as outf:
                    outf.write('\n'.join([t + '\t1' for t in targets]))

def export_krapivin_maui():
    # prepare logging.
    config  = keyphrase.config.setup_keyphrase_all()   # load settings.

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config)

    # keep the first 400 in krapivin
    dataset = test_sets['krapivin']

    train_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/krapivin/train/'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    train_texts = dataset['source_str'][401:]
    train_targets = dataset['target_str'][401:]
    for i, (train_text, train_target) in enumerate(zip(train_texts,train_targets)):
        print('train '+ str(i))
        with open(train_dir+str(i)+'.txt', 'w') as f:
            f.write(' '.join(train_text))
        with open(train_dir + str(i) + '.key', 'w') as f:
            f.write('\n'.join([' '.join(t)+'\t1' for t in train_target]))

    test_dir  = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/krapivin/test/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_texts = dataset['source_str'][:400]
    test_targets = dataset['target_str'][:400]
    for i, (test_text, test_target) in enumerate(zip(test_texts,test_targets)):
        print('test '+ str(i))
        with open(test_dir+str(i)+'.txt', 'w') as f:
            f.write(' '.join(test_text))
        with open(test_dir + str(i) + '.key', 'w') as f:
            f.write('\n'.join([' '.join(t)+'\t1' for t in test_target]))

def export_ke20k_testing_maui():
    from keyphrase.dataset import keyphrase_test_dataset
    target_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/ke20k/'

    config  = keyphrase.config.setup_keyphrase_all()   # load settings.
    doc_list = keyphrase_test_dataset.testing_data_loader('ke20k', kwargs=dict(basedir = config['path'])).get_docs(False)

    for d in doc_list:
        d_id = d.name[:d.name.find('.txt')]
        print(d_id)
        with open(target_dir+d_id+'.txt', 'w') as textfile:
            textfile.write(d.title+'\n'+d.text)
        with open(target_dir + d_id + '.key', 'w') as phrasefile:
            for p in d.phrases:
                phrasefile.write('%s\t1\n' % p)

def export_ke20k_train_maui():
    '''
    just use the validation dataset
    :return:
    '''
    config  = keyphrase.config.setup_keyphrase_all()   # load settings.
    target_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/ke20k/train/'

    import emolga,string

    printable = set(string.printable)
    validation_records = emolga.dataset.build_dataset.deserialize_from_file(config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'validation_record_'+str(config['validation_size'])+'.pkl')
    for r_id, r in enumerate(validation_records):
        print(r_id)

        r['title'] = filter(lambda x: x in printable, r['title'])
        r['abstract'] = filter(lambda x: x in printable, r['abstract'])
        r['keyword'] = filter(lambda x: x in printable, r['keyword'])

        with open(target_dir+str(r_id)+'.txt', 'w') as textfile:
            textfile.write(r['title']+'\n'+r['abstract'])

        with open(target_dir + str(r_id) + '.key', 'w') as phrasefile:
            for p in r['keyword'].split(';'):
                phrasefile.write('%s\t1\n' % p)

def prepare_data_cross_validation(input_dir, output_dir, folds=5):
    file_names = [ w[:w.index('.')] for w in filter(lambda x: x.endswith('.txt'),os.listdir(input_dir))]
    file_names.sort()
    file_names = numpy.asarray(file_names)

    fold_size = len(file_names)/folds

    for fold in range(folds):
        start   = fold * fold_size
        end     = start + fold_size

        if (fold == folds-1):
            end = len(file_names)

        print('Fold %d' % fold)

        test_names = file_names[start: end]
        train_names = file_names[list(filter(lambda x: x < start or x >= end, range(len(file_names))))]
        # print('test_names: %s' % str(test_names))
        # print('train_names: %s' % str(train_names))

        train_dir = output_dir + 'train_'+str(fold+1)+'/'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = output_dir + 'test_'+str(fold+1)+'/'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for test_name in test_names:
            shutil.copyfile(input_dir + test_name + '.txt', test_dir + test_name + '.txt')
            shutil.copyfile(input_dir + test_name + '.key', test_dir + test_name + '.key')
        for train_name in train_names:
            shutil.copyfile(input_dir + test_name + '.txt', train_dir + train_name + '.txt')
            shutil.copyfile(input_dir + test_name + '.key', train_dir + train_name + '.key')

if __name__ == '__main__':
    # export_krapivin_maui()
    # export_maui()
    # input_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/ke20k/'
    # output_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/ke20k/cross_validation/'
    # prepare_data_cross_validation(input_dir, output_dir, folds=5)

    # export_ke20k_maui()
    export_ke20k_train_maui()