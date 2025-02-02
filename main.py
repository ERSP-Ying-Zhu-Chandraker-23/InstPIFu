import argparse
from configs.config_utils import CONFIG

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Refer-it-in-RGBD training')
    parser.add_argument('--config', type=str, default='config/pretrian_config.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    cfg=CONFIG(args.config)
    cfg.update_config(args.__dict__)

    cfg.log_string('Loading configuration')
    cfg.log_string(cfg.config)
    cfg.write_config()
    '''
    Run
    '''
    if cfg.config['mode']=='train':
        import train
        train.run(cfg)
    if cfg.config['mode']=='demo':
        import demo
        demo.run(cfg)
    elif cfg.config['mode']=='test':
        import test
        test.run(cfg)