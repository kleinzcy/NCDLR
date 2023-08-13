import os
import logging
import random



def get_net_builder(net_name, from_name: bool):
    """
    built network according to network name
    return **class** of backbone network (not instance).

    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
    """
    if from_name:
        import torchvision.models as nets
        model_name_list = sorted(name for name in nets.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(nets.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return nets.__dict__[net_name]
    else:
        import nets as nets
        builder = getattr(nets, net_name)
        return builder



def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger



def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()
