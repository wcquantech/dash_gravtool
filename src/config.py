import torchvision.transforms as transform

def create_config(cfg, batch_size, num_workers, test_set_path, shuffle, num_models, models):
    cfg['batch_size'] = batch_size
    cfg['num_workers'] = num_workers
    cfg['test_set_path'] = test_set_path
    cfg['shuffle'] = shuffle
    cfg['pin_memory'] = True
    cfg['num_models'] = num_models
    cfg['models'] = models
    cfg['transforms'] = {}
    for model in models:
        if "inception" in model.lower():
            cfg['transforms']["inception-v3_checkpoint"] = transform.Compose([
                                            transform.Resize([299, 299]),
                                            transform.ToTensor(),
                                            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        if "google" in model.lower():
            cfg['transforms']["google-net_checkpoint"] = transform.Compose([
                                            transform.Resize([224, 224]),
                                            transform.ToTensor(),
                                            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        if "mnist" in model.lower():
            cfg['transforms']["mnist_model"] = transform.Compose(
                                                [transform.ToTensor(),
                                                 transform.Grayscale(),
                                                 transform.Normalize((0.5,), (0.5,))
                                                 ])


    print(cfg)
    return cfg



def create_config_single_model(cfg_single, model_path):
    cfg_single['model'] = model_path
    if "inception" in cfg_single['model'].lower():
        cfg_single['transforms'] = transform.Compose([
                                            transform.Resize([299, 299]),
                                            transform.ToTensor(),
                                            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
    elif "google" in cfg_single['model'].lower():
        cfg_single['transforms'] = transform.Compose([
                                            transform.Resize([224, 224]),
                                            transform.ToTensor(),
                                            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
    return cfg_single




#
# cfg = {}
# cfg['batch_size'] = 8
# cfg['num_workers'] = 4
# cfg['test_set_path'] = r'/Users/chun/Documents/2022/WEBTOOL/test'
# cfg['shuffle'] = False
# cfg['pin_memory'] = True
# cfg['num_models'] = 2
# cfg['need_confusion_matrix'] = True
# cfg['models'] = {"Inception-v3": r'/Users/chun/Documents/2022/WEBTOOL/inception-v3_checkpoint.pt',
#                  "GoogLeNet": r'/Users/chun/Documents/2022/WEBTOOL/google-net_checkpoint.pt'
#                  }
#
# cfg['transforms'] = {"Inception-v3": transforms.Compose([
#                                     transforms.Resize([299,299]),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                     ]),
#                         "GoogLeNet": transforms.Compose([
#                                     transforms.Resize([224,224]),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                     ])}