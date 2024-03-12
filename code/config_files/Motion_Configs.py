class Config(object):
    def __init__(self):
        # model configs for TFC 
        self.input_channels = 8  # feature count
        self.transformer_nhead = 2
        self.transformer_num_layers = 2
        self.TSlength_aligned = 900  # sequence length 15Hz * 60second
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4  # original lr: 3e-4

        # model configs for gpt4ts 
        self.patch_size = 32 
        self.stride = 16 
        self.d_model = 768  # this is determined by gpt model. 
        self.dropout = 0.1
        self.gpt_layers = 6
        self.feat_dim = 8  # feature count
        self.max_seq_len = 900  # seq lenth 
        self.optimizer = "RAdam"
        self.gpt_lr = 0.001

        # common configs 
        self.embedding_len = 160  # final embedding len = embedding_len*2
        self.num_epoch = 2
        self.batch_size = 2 
        self.drop_last = True
        
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True
        self.use_cosine_similarity_f = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10