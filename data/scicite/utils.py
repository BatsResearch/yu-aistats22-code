import json

label_idx = ['background', 'method', 'result']
def load_jsonl(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_item(data, field_name):
    result = []
    for item in data:
        result.append(item[field_name])
    return result

def get_inst(data, keep_instance=False):
    strings = []
    label = []
    for inst in data:
        strings.append(inst['string'].lower())
        label.append(label_idx.index(inst['label']))
    if keep_instance:
        return data, label
    return strings, label

def textsplit(instance, p=False):
    string = instance['string']
    try:
        pretext = string[:int(instance['citeStart'])]
        postext = string[int(instance['citeEnd']):]
        mid = string[int(instance['citeStart']):int(instance['citeEnd'])]
    except ValueError:
        pretext=mid=postext = ''
    if p:
        print('-'*50)
        print(pretext)
        print(mid)
        print(postext)
    return pretext, mid, postext