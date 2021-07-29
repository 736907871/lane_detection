import os

#####################################
#####################################
#用来合并train_set里的.json合并在一起
#####################################
#####################################


root_dir = '../../data/tusimple/train_set'
src_json = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
infos = []
dst_json = root_dir + '/test_label.json'

for json_file in src_json:
    src_path = os.path.join(root_dir, json_file)
    with open(src_path, 'r') as f:
        infos += f.readlines()
if os.path.exists(dst_json):
    os.remove(dst_json)
with open(dst_json, 'a') as f_dst:
    for ele in infos:
        f_dst.write(ele + '\n')
