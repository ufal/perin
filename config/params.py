import yaml
import copy


class Params:
    def __init__(self):
        self.accumulation_steps = 2                  # number of gradient accumulation steps for achieving a bigger batch_size
        self.activation = "relu"                     # transformer (decoder) activation function, supported values: {'relu', 'gelu', 'sigmoid', 'mish'}
        self.balance_loss_weights = True             # use weight loss balancing (GradNorm)
        self.batch_size = 16                         # batch size (further divided into multiple GPUs)
        self.beta_2 = 0.98                           # beta 2 parameter for Adam(W) optimizer
        self.blank_weight = 1.0                      # weight of cross-entropy loss for predicting an empty label
        self.char_embedding = True                   # use character embedding in addition to bert
        self.char_embedding_size = 128               # dimension of the character embedding layer in the character embedding module
        self.decoder_delay_steps = 0                 # number of initial steps with frozen decoder
        self.decoder_learning_rate = 6e-4            # initial decoder learning rate
        self.decoder_weight_decay = 1.2e-6           # amount of weight decay
        self.dropout_anchor = 0.5                    # dropout at the last layer of anchor classifier
        self.dropout_edge_label = 0.5                # dropout at the last layer of edge label classifier
        self.dropout_edge_presence = 0.5             # dropout at the last layer of edge presence classifier
        self.dropout_edge_attribute = 0.5            # dropout at the last layer of edge presence classifier
        self.dropout_label = 0.5                     # dropout at the last layer of label classifier
        self.dropout_property = 0.7                  # dropout at the last layer of property classifier
        self.dropout_top = 0.9                       # dropout at the last layer of top classifier
        self.dropout_transformer = 0.1               # dropout for the transformer layers (decoder)
        self.dropout_transformer_attention = 0.1     # dropout for the transformer's attention (decoder)
        self.dropout_word = 0.1                      # probability of dropping out a whole word from the encoder (in favour of char embedding)
        self.encoder = "xlm-roberta-base"            # pretrained encoder model
        self.encoder_delay_steps = 2000              # number of initial steps with frozen XLM-R
        self.encoder_freeze_embedding = True         # freeze the first embedding layer in XLM-R
        self.encoder_learning_rate = 6e-5            # initial encoder learning rate
        self.encoder_weight_decay = 1e-2             # amount of weight decay
        self.epochs = 100                            # number of epochs for train
        self.focal = True                            # use focal loss for the label prediction
        self.grad_norm_alpha = 1.5                   # grad-norm sensitivity
        self.grad_norm_lr = 1e-3                     # learning rate for the grad-norm optimizer
        self.group_ops = False                       # group 'opN' edge labels into one
        self.hidden_size_ff = 4 * 768                # hidden size of the transformer feed-forward submodule
        self.hidden_size_anchor = 128                # hidden size anchor biaffine layer
        self.hidden_size_edge_label = 256            # hidden size for edge label biaffine layer
        self.hidden_size_edge_presence = 512         # hidden size for edge label biaffine layer
        self.hidden_size_edge_attribute = 128        # hidden size for edge label biaffine layer
        self.label_smoothing = 0.1                   # amount of label smoothing applied for label classification
        self.layerwise_lr_decay = 1.0                # layerwise decay of learning rate in the encoder
        self.n_attention_heads = 8                   # number of attention heads in the decoding transformer
        self.n_layers = 3                            # number of layers in the decoder
        self.n_mixture_components = 15               # number of components in the mixture of softmaxes for the label output
        self.normalize = True                        # normalize inverted edge directions and labels
        self.query_length = 4                        # number of queries genereted for each word on the input
        self.pre_norm = True                         # use pre-normalized version of the transformer (as in Transformers without Tears)
        self.warmup_steps = 6000                     # number of the warm-up steps for the inverse_sqrt scheduler

    def init_data_paths(self, base_dir: str):
        # path to the training dataset
        self.training_data = {
            ("amr", "eng"): f"{base_dir}/2020/cf/training/amr.mrp",
            ("amr", "zho"): f"{base_dir}/2020/cl/training/amr.zho_train.mrp",
            ("drg", "eng"): f"{base_dir}/2020/cf/training/drg.mrp",
            ("drg", "deu"): f"{base_dir}/2020/cl/training/drg.deu_train.mrp",
            ("eds", "eng"): f"{base_dir}/2020/cf/training/eds.mrp",
            ("ptg", "eng"): f"{base_dir}/2020/cf/training/ptg.mrp",
            ("ptg", "ces"): f"{base_dir}/2020/cl/training/ptg.ces_train.mrp",
            ("ucca", "eng"): f"{base_dir}/2020/cf/training/ucca.mrp",
            ("ucca", "deu"): f"{base_dir}/2020/cl/training/ucca.deu_train.mrp",
        }

        # path to the validation dataset
        self.validation_data = {
            ("amr", "eng"): f"{base_dir}/2020/cf/validation/amr.mrp",
            ("amr", "zho"): f"{base_dir}/2020/cl/training/amr.zho_val.mrp",
            ("drg", "eng"): f"{base_dir}/2020/cf/validation/drg.mrp",
            ("drg", "deu"): f"{base_dir}/2020/cl/training/drg.deu_val.mrp",
            ("eds", "eng"): f"{base_dir}/2020/cf/validation/eds.mrp",
            ("ptg", "eng"): f"{base_dir}/2020/cf/validation/ptg.mrp",
            ("ptg", "ces"): f"{base_dir}/2020/cl/training/ptg.ces_val.mrp",
            ("ucca", "eng"): f"{base_dir}/2020/cf/validation/ucca.mrp",
            ("ucca", "deu"): f"{base_dir}/2020/cl/training/ucca.deu_val.mrp",
        }

        # path to the test dataset
        self.test_data = {
            ("amr", "eng"): f"{base_dir}/2020/cf/evaluation/input.mrp",
            ("amr", "zho"): f"{base_dir}/2020/cl/evaluation/input.mrp",
            ("drg", "eng"): f"{base_dir}/2020/cf/evaluation/input.mrp",
            ("drg", "deu"): f"{base_dir}/2020/cl/evaluation/input.mrp",
            ("eds", "eng"): f"{base_dir}/2020/cf/evaluation/input.mrp",
            ("ptg", "eng"): f"{base_dir}/2020/cf/evaluation/input.mrp",
            ("ptg", "ces"): f"{base_dir}/2020/cl/evaluation/input.mrp",
            ("ucca", "eng"): f"{base_dir}/2020/cf/evaluation/input.mrp",
            ("ucca", "deu"): f"{base_dir}/2020/cl/evaluation/input.mrp",
        }

        # path to udpipe companion data
        self.companion_data = {
            ("amr", "eng"): f"{base_dir}/2020/cf/companion/combined_udpipe.mrp",
            ("amr", "zho"): f"{base_dir}/2020/cl/companion/combined_zho.mrp",
            ("drg", "eng"): f"{base_dir}/2020/cf/companion/combined_udpipe.mrp",
            ("drg", "deu"): f"{base_dir}/2020/cl/companion/combined_deu_translated.mrp",
            ("eds", "eng"): f"{base_dir}/2020/cf/companion/combined_udpipe.mrp",
            ("ptg", "eng"): f"{base_dir}/2020/cf/companion/combined_udpipe.mrp",
            ("ptg", "ces"): f"{base_dir}/2020/cl/companion/combined_ces.mrp",
            ("ucca", "eng"): f"{base_dir}/2020/cf/companion/combined_udpipe.mrp",
            ("ucca", "deu"): f"{base_dir}/2020/cl/companion/combined_deu.mrp",
        }

        return self

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        return self

    def state_dict(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return {k: self.__dict__[k] for k in members}

    def load(self, args):
        self.init_data_paths(args.data_directory)
        with open(args.config, "r", encoding="utf-8") as f:
            params = yaml.load(f)
            self.load_state_dict(params)

    def save(self, json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            d = self.state_dict()
            yaml.dump(d, f)

    def get_hyperparameters(self):
        clone = copy.copy(self)
        del clone.training_data
        del clone.validation_data
        del clone.test_data
        del clone.companion_data
        del clone.frameworks
        return clone
