import requests
import re

from pprint import pprint

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def print_parameter_count(model, is_simplify=False, is_print_all=False, is_print_detail=False,contain_str=None):
    regex = re.compile("(\\.)")
    params = model.named_parameters()
    select_parameters_count = 0
    if contain_str != None:
        params = filter(lambda it: contain_str in it[0], params)
    if is_simplify:
        print("total_parameters_count: %d" % sum(p.numel() for p in model.parameters()))
        print("train_parameters_count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
        if is_print_all:
            for name, param in params:
                select_parameters_count = select_parameters_count + param.numel()
                if is_print_detail:
                    print(name + "\t\t\t\t\t\t\t\t" + str(param.numel()))
                    continue
                if len(list(regex.finditer(name))) > 1:
                    index = list(regex.finditer(name))[-2].start()
                else:
                    index = 0
                print(name[index:] + "\t\t\t\t\t\t\t\t" + str(param.numel()))
            if contain_str != None:
                print("select_parameters_count: %d" % select_parameters_count)
    else:
        count = {}
        count["total_parameters_count"] = sum(p.numel() for p in model.parameters())
        count["train_parameters_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_print_all:
            for name, param in params:
                select_parameters_count = select_parameters_count + param.numel()
                if is_print_detail:
                    count[name] = param.numel()
                    continue
                if len(list(regex.finditer(name))) > 1:
                    index = list(regex.finditer(name))[-2].start()
                else:
                    index = 0
                count[name[index:]] = param.numel()
            if contain_str != None:
                print("select_parameters_count: %d" % select_parameters_count)
        pprint(count)