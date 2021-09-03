import json
import codecs

# writh json
def write_json(_dataset, _path, intent=4, separate_store=False):
    if separate_store:
        for data_idx, data in enumerate(_dataset):
            write_json(data, _path+'_'+str(data_idx)+'.json', intent)
    else:
        with codecs.open(_path, 'w', 'utf-8') as file_write:
            json.dump(_dataset, file_write, indent=intent)
            file_write.close()

# read json
def read_json(_path, file_num=0):
    if _path.endswith('json'):
        with codecs.open(_path, 'r', 'utf-8') as file_read:
            _dataset = json.load(file_read)
            file_read.close()
    else:
        _dataset = [read_json(_path+'_'+str(i)+'.json')  for i in range(file_num)]
    return _dataset