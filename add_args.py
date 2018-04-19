def add_args(args, parser):
    for d in args:
        if 'nargs' not in d:
            d['nargs'] = None
        if 'required' not in d:
            d['required'] = False
        parser.add_argument('--'+d['name'],
                            nargs=d['nargs'],
                            default=d['default'],
                            type=d['type'],
                            help=d['help'],
                            required=d['required'])


def for_general(parser):
    args = [
    {
     'name': 'data_dir',
     'default': 'data',
     'type': str,
     'help': 'folder of dataset'},
    {
     'name': 'model_dir',
     'default': None,
     'type': str,
     'help': 'folder from which restore the model '},
    {
     'name': 'image_size',
     'nargs': 2,
     'default': [576,1024],
     'type': int,
     'help': 'Size of the input image'},
    {
     'name': 'gazemap_size',
     'nargs': 2,
     'default': [36,64],
     'type': int,
     'help': 'Size of the predicted gaze map'}
    ]
    add_args(args, parser)


def for_inference(parser):
    args = [
    {
     'name': 'batch_size',
     'default': 20,
     'type': int,
     'help': 'basic batch size'},
    {
     'name': 'use_prior',
     'default': False,
     'type': bool,
     'help': 'whether to use prior gaze map'},
    {
     'name': 'drop_rate',
     'default': 0,
     'type': float,
     'help': 'drop rate'},
    {
     'name': 'readout',
     'default': 'default',
     'type': str,
     'help': 'which readout network to use'},
    {
     'name': 'sparsity_weight',
     'default': 0,
     'type': float,
     'help': 'The weight of sparsity regularization'}, 
    {
     'name': 'gpu_memory_fraction',
     'default': None,
     'type': float,
     'help': 'The fraction of GPU memory to use'},
     {
     'name': 'binary',
     'default': False,
     'type': bool,
     'help': 'Whether to make the gaze maps to binary maps'},
     {
     'name': 'annotation_threshold',
     'default': None,
     'type': float,
     'help': 'When the gaze density is more than annotation_threshold times the uniform density, the pixel is gazed'}
    ]
    add_args(args, parser)
    
    
def for_feature(parser):
    args = [
    {
     'name': 'feature_name',
     'default': 'alexnet',
     'type': str,
     'help': 'Which kind of features to use'},
    {
     'name': 'feature_map_size',
     'nargs': 2,
     'default': [36, 64],
     'type': int,
     'help': 'Feature map size (not include the number of channels)'},
    {
     'name': 'feature_map_channels',
     'default': 2560,
     'type': int,
     'help': 'The number of feature map channels'}
    ]
    add_args(args, parser)
    
    
def for_full(parser):
    args = [
    {
     'name': 'encoder',
     'default': 'vgg',
     'type': str,
     'help': 'Which encoder to use'}
    ]
    add_args(args, parser)
    
    
def for_training(parser):
    args = [
    {
     'name': 'learning_rate',
     'default': 1e-3,
     'type': float,
     'help': 'Learning rate for Adam Optimizer'},
    {
     'name': 'max_iteration',
     'default': 10001,
     'type': int,
     'help': 'Maximum iterations'},
    {
     'name': 'train_epochs',
     'default': 10,
     'type': int,
     'help': 'For how many epochs the model should be trained in total'},
    {
     'name': 'epochs_before_validation',
     'default': 1,
     'type': int,
     'help': 'For how many epochs the model should be trained before each time of validation'},
    {
     'name': 'quick_summary_period',
     'default': 10,
     'type': int,
     'help': 'After how many iterations do some quick summaries'},
    {
     'name': 'slow_summary_period',
     'default': 50,
     'type': int,
     'help': 'After how many iterations do some slow summaries'},
    {
     'name': 'valid_summary_period',
     'default': 500,
     'type': int,
     'help': 'After how many iterations do validation and save one checkpoint'},
    {
     'name': 'valid_batch_factor',
     'default': 2,
     'type': int,
     'help': 'The batch size for validation is equal to this number multiply the original batch size'},
    {
     'name': 'logs_dir',
     'default': None,
     'type': str,
     'help': 'path to logs directory'},
    {
     'name': 'weight_data',
     'default': False,
     'type': bool,
     'help': 'whether to weight the data points differently in trianing'}
    ]
    add_args(args, parser)
    
    
def for_evaluation(parser):
    args = [
    {
     'name': 'model_iteration',
     'default': None,
     'type': str,
     'help': 'The model of which iteration to resotre'}
    ]
    add_args(args, parser)
    
    
def for_visualization(parser):
    args = [
    {
     'name': 'model_iteration',
     'default': None,
     'type': str,
     'help': 'The model of which iteration to restore'},
    {
     'name': 'visualization_thresh',
     'default': 1e-5,
     'type': float,
     'help': 'Probability density threshold for visualization'},
    {
     'name': 'video_list_file',
     'default': None,
     'type': str,
     'help': 'A txt file that contains the list of the videos to visualize, seperated by space'},
    {
     'name': 'fps',
     'default': 3,
     'type': float,
     'help': 'Frames per second'},
    {
     'name': 'heatmap_alpha',
     'default': 0.5,
     'type': float,
     'help': 'Transparency for heat map. 1 is fully opaque.'},
    {
     'name': 'turing_area_table',
     'default': None,
     'type': str,
     'help': 'Path to the table that stores the highlighted areas of Turing GT videos.'},
    {
     'name': 'skip_first_n_frames',
     'default': None,
     'type': int,
     'help': 'Number of frames to skip in the beginning.'}  
    ]
    add_args(args, parser)


def for_lstm(parser):
    args = [
    {
     'name': 'n_steps',
     'default': None,
     'type': int,
     'help': 'number of time steps for each sequence'},
     {
     'name': 'longest_seq',
     'default': None,
     'type': int,
     'help': 'How many frames can the longest sequence contain'}
    ]
    add_args(args, parser)
