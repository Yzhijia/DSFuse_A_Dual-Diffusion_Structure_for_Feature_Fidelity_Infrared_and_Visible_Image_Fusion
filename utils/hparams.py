import argparse
import os
import yaml

global_print_hparams = True
hparams = {}
opt = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(config='', exp_name='', hparams_str='', print_hparams=True, global_hparams=True):
    if config == '' and exp_name == '':
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('-hp', '--hparams', type=str, default='',
                            help='location of the data corpus')


        # 网络超参
        parser.add_argument('--fusion_loss_type', type=str, default='SADFusion', help='SADFusion loss')
        parser.add_argument('--fusion_model_type', type=str, default="RTFusionNet", help='fusion_model_structure')
        parser.add_argument('--p1', type=int, default=1, help='ir pixel')
        parser.add_argument('--p2', type=int, default=1, help='vi pixel')
        parser.add_argument('--p3', type=int, default=1, help='ir gradient')
        parser.add_argument('--p4', type=int, default=1, help='vi gradient')
        parser.add_argument('--p5', type=int, default=1, help='max gradient')
        parser.add_argument('--p6', type=int, default=1, help='ssim')
        parser.add_argument('--p_diff', type=int, default=5, help='diffusion_loss')
        parser.add_argument('--res_type', type=bool, default=False, help='loss_prediction_type')
        

        # parser.add_argument('--fusion_loss_type', type=str, default='SADFusion', help='exp_name')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--debug', action='store_true', help='debug')
        


        args, unknown = parser.parse_known_args()
        print("| Unknow hparams: ", unknown)
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, validate=False, reset=False, debug=False)
    global hparams
    global opt
    assert args.config != '' or args.exp_name != ''
    if args.config != '':
        assert os.path.exists(args.config)

    config_chains = []
    loaded_config = set()

    def load_config(config_fn):
        # deep first inheritance and avoid the second visit of one node
        if not os.path.exists(config_fn):
            return {}
        with open(config_fn, encoding="utf-8") as f:
            hparams_ = yaml.safe_load(f)
        loaded_config.add(config_fn)
        if 'base_config' in hparams_:
            ret_hparams = {}
            if not isinstance(hparams_['base_config'], list):
                hparams_['base_config'] = [hparams_['base_config']]
            for c in hparams_['base_config']:
                if c.startswith('.'):
                    c = f'{os.path.dirname(config_fn)}/{c}'
                    c = os.path.normpath(c)
                if c not in loaded_config:
                    override_config(ret_hparams, load_config(c))
            override_config(ret_hparams, hparams_)
        else:
            ret_hparams = hparams_
        config_chains.append(config_fn)
        return ret_hparams

    saved_hparams = {}
    args_work_dir = ''
    if args.exp_name != '':
        args_work_dir = f'//data1/yzj/checkpoint/transfusion_diffusion/{args.exp_name}'
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                saved_hparams_ = yaml.safe_load(f)
                if saved_hparams_ is not None:
                    saved_hparams.update(saved_hparams_)
    hparams_ = {}
    if args.config != '':
        hparams_.update(load_config(args.config))
    if not args.reset:
        hparams_.update(saved_hparams)
    hparams_['work_dir'] = args_work_dir

    # 网络超参赋值
    hparams_['p1'] = args.p1
    hparams_['p2'] = args.p2
    hparams_['p3'] = args.p3
    hparams_['p4'] = args.p4
    hparams_['p5'] = args.p5
    hparams_['p6'] = args.p6
    hparams_['p_diff'] = args.p_diff
    hparams_['fusion_loss_type'] = args.fusion_loss_type
    hparams_['fusion_model_type'] = args.fusion_model_type
    hparams_['res_type'] = args.res_type
    # Support config overriding in command line. Support list type config overriding.
    # Examples: --hparams="a=1,b.c=2,d=[1 1 1]"
    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            v = v.strip("\'\" ")
            config_node = hparams_
            for k_ in k.split(".")[:-1]:
                config_node = config_node[k_]
            k = k.split(".")[-1]
            if k not in config_node:
                config_node[k] = v

            elif v in ['True', 'False'] or type(config_node[k]) in [bool, list, dict]:
                if type(config_node[k]) == list:
                    v = v.replace(" ", ",")
                config_node[k] = eval(v)
            else:
                config_node[k] = type(config_node[k])(v)
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
        os.makedirs(hparams_['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams_, f)

    hparams_['infer'] = args.infer
    hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    hparams_['exp_name'] = args.exp_name
    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    if print_hparams and global_print_hparams and global_hparams:
        print('| Hparams chains: ', config_chains)
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(hparams_.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        global_print_hparams = False
    return hparams_
