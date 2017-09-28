import time

__author__ = 'jiataogu'
import os
import os.path as path


def setup_keyphrase_all():
    config = dict()
    '''
    Meta information
    '''
    config['seed']            = 154316847
    # for naming the outputs and logs
    config['model_name']      = 'CopyRNN' # 'TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'Maui', 'Kea', 'RNN', 'CopyRNN'
    config['task_name']       = 'keyphrase-all.copy'
    config['timemark']        = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    config['path']            = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) #path.realpath(path.curdir)
    config['path_experiment'] = config['path'] + '/Experiment/'+config['task_name']
    config['path_h5']         = config['path_experiment']
    config['path_log']        = config['path_experiment']

    config['casestudy_log']   = config['path_experiment'] + '/case-print.log'

    '''
    Experiment process
    '''
    # do training?
    config['do_train']        = True
    # config['do_train']        = False

    # do quick-testing (while training)?
    config['do_quick_testing']     = True
    # config['do_quick_testing']     = False

    # do validation?
    config['do_validate']     = True
    # config['do_validate']     = False

    # do predicting?
    config['do_predict']      = True
    # config['do_predict']      = False

    # do testing?
    config['do_evaluate']     = True
    # config['do_evaluate']     = False

    '''
    Training settings
    '''
    # Dataset
    config['training_name']   = 'acm-sci-journal_600k'

    # actually still not clean enough, further filtering is done when loading pairs: dataset_utils.load_pairs()
    config['training_dataset']= config['path'] + '/dataset/keyphrase/million-paper/all_title_abstract_keyword_clean.json'
    # config['testing_name']    = 'inspec_all'
    # config['testing_dataset'] = config['path'] + '/dataset/keyphrase/inspec/inspec_all.json'

    config['testing_datasets']= ['nus'] # 'inspec', 'nus', 'semeval', 'krapivin', 'kp20k'
    config['preprocess_type'] = 1 # 0 is old type, 1 is new type(keep most punctuation)

    config['data_process_name'] = 'punctuation-20000validation-20000testing/'

    config['validation_size'] = 20000
    config['validation_id']   = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'validation_id_'+str(config['validation_size'])+'.pkl'
    config['testing_id']      = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'testing_id_'+str(config['validation_size'])+'.pkl'
    config['dataset']         = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'all_600k_dataset.pkl'
    config['voc']             = config['path'] + '/dataset/keyphrase/'+config['data_process_name']+'all_600k_voc.pkl' # for manual check

    # Optimization
    config['use_noise']       = False
    config['optimizer']       = 'adam'
    config['clipnorm']        = 0.1

    config['save_updates']    = True
    config['get_instance']    = True

    # size
    config['batch_size']      = 50
    # config['mini_batch_size'] = 20 # not useful any more
    config['mini_mini_batch_length']      = 300000 # max length (#words) of each mini-mini batch, up to the GPU memory you have
    config['mode']            = 'RNN'
    config['binary']          = False
    config['voc_size']        = 50000

    # output log place
    if not os.path.exists(config['path_log']):
        os.mkdir(config['path_log'])

    # path to pre-trained model
    config['trained_model']   = '' #config['path_experiment'] + '/experiments.keyphrase-all.one2one.copy.id=20170106-025508.epoch=4.batch=1000.pkl'
    # config['trained_model']   = config['path_experiment'] + '/experiments.keyphrase-all.one2one.copy.id=20170106-025508.epoch=4.batch=1000.pkl'

    config['weight_json']= config['path_experiment'] + '/model_weight.json'
    config['resume_training'] = False
    config['training_archive']= None

    '''
    Predicting/evaluation settings
    '''
    config['baseline_data_path']     = config['path'] + '/dataset/keyphrase/baseline-data/'
    # whether to add length penalty on beam search results
    config['normalize_score']   = False
    # whether to keep the longest prediction when many phrases sharing same prefix, like for 'A','AB','ABC' we only keep 'ABC'
    config['keep_longest']      = False

    # whether do filtering on groundtruth? 'appear-only','non-appear-only' and None (do no filtering)
    config['target_filter']     = 'appear-only'
    # whether do filtering on predictions? 'appear-only','non-appear-only' and None (do no filtering)
    config['predict_filter']    = 'appear-only'

    config['noun_phrase_only']  = False

    config['max_len']         = 6
    config['sample_beam']     = 200 #config['voc_size']
    config['sample_stoch']    = False # use beamsearch
    config['sample_argmax']   = False
    config['return_encoding'] = False

    config['predict_type']    = 'generative' # type of prediction, extractive or generative
    # config['predict_path']    = config['path_experiment'] + '/predict.' + config['predict_type']+ '.'+ config['timemark'] + '.dataset=%d.len=%d.beam=%d.predict=%s.target=%s.keeplongest=%s.noun_phrase=%s/' % (len(config['testing_datasets']),config['max_len'], config['sample_beam'], config['predict_filter'], config['target_filter'], config['keep_longest'], config['noun_phrase_only'])
    config['predict_path']      = os.path.join(config['path_experiment'], 'predict.generative.20170712-221404.dataset=1.len=6.beam=200.predict=appear-only.target=appear-only.keeplongest=False.noun_phrase=False/')

    if not os.path.exists(config['predict_path']):
        os.mkdir(config['predict_path'])

    '''
    Model settings
    '''
    # Encoder: Model
    config['bidirectional']   = True
    config['enc_use_contxt']  = False
    config['enc_learn_nrm']   = True
    config['enc_embedd_dim']  = 150    # 100
    config['enc_hidden_dim']  = 300    # 150
    config['enc_contxt_dim']  = 0
    config['encoder']         = 'RNN'
    config['pooling']         = False

    # Decoder: dimension
    config['dec_embedd_dim']  = 150  # 100
    config['dec_hidden_dim']  = 300  # 180
    config['dec_contxt_dim']  = config['enc_hidden_dim']       \
                                if not config['bidirectional'] \
                                else 2 * config['enc_hidden_dim']

    # Decoder: CopyNet
    config['copynet']         = True
    # config['copynet']         = False
    config['identity']        = False
    config['location_embed']  = True
    config['coverage']        = True
    config['copygate']        = False

    # Decoder: Model
    config['shared_embed']    = False
    config['use_input']       = True
    config['bias_code']       = True
    config['dec_use_contxt']  = True
    config['deep_out']        = False
    config['deep_out_activ']  = 'tanh'  # maxout2
    config['bigram_predict']  = True
    config['context_predict'] = True
    config['dropout']         = 0.5  # 5
    config['leaky_predict']   = False

    config['dec_readout_dim'] = config['dec_hidden_dim']
    if config['dec_use_contxt']:
        config['dec_readout_dim'] += config['dec_contxt_dim']
    if config['bigram_predict']:
        config['dec_readout_dim'] += config['dec_embedd_dim']

    # Decoder: sampling
    config['multi_output']    = False

    config['decode_unk']      = False
    config['explicit_loc']    = False

    # Gradient Tracking !!!
    config['gradient_check']  = True
    config['gradient_noise']  = True

    config['skip_size']       = 15

    # for w in config:
    #     print('{0} => {1}'.format(w, config[w]))
    # print('setup ok.')
    return config
