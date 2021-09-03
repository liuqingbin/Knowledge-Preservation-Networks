import argparse
import os
from utils.basic_func import read_json, write_json

train_file_sever = ['dialogues_011.json', 'dialogues_007.json', 'dialogues_027.json', 'dialogues_087.json',
                    'dialogues_085.json', 'dialogues_127.json', 'dialogues_106.json', 'dialogues_026.json',
                    'dialogues_113.json', 'dialogues_022.json', 'dialogues_094.json', 'dialogues_048.json',
                    'dialogues_055.json', 'dialogues_121.json', 'dialogues_123.json', 'dialogues_018.json',
                    'schema.json',
                    'dialogues_066.json', 'dialogues_049.json', 'dialogues_032.json', 'dialogues_122.json',
                    'dialogues_068.json', 'dialogues_067.json', 'dialogues_104.json', 'dialogues_071.json',
                    'dialogues_059.json', 'dialogues_069.json', 'dialogues_126.json', 'dialogues_060.json',
                    'dialogues_058.json', 'dialogues_065.json', 'dialogues_103.json', 'dialogues_016.json',
                    'dialogues_118.json', 'dialogues_003.json', 'dialogues_029.json', 'dialogues_046.json',
                    'dialogues_096.json', 'dialogues_098.json', 'dialogues_114.json', 'dialogues_107.json',
                    'dialogues_111.json', 'dialogues_076.json', 'dialogues_062.json', 'dialogues_120.json',
                    'dialogues_072.json', 'dialogues_110.json', 'dialogues_063.json', 'dialogues_001.json',
                    'dialogues_014.json', 'dialogues_084.json', 'dialogues_090.json', 'dialogues_037.json',
                    'dialogues_040.json', 'dialogues_061.json', 'dialogues_054.json', 'dialogues_112.json',
                    'dialogues_089.json', 'dialogues_030.json', 'dialogues_045.json', 'dialogues_013.json',
                    'dialogues_092.json', 'dialogues_036.json', 'dialogues_025.json', 'dialogues_075.json',
                    'dialogues_044.json', 'dialogues_041.json', 'dialogues_031.json', 'dialogues_078.json',
                    'dialogues_033.json', 'dialogues_002.json', 'dialogues_010.json', 'dialogues_023.json',
                    'dialogues_100.json', 'dialogues_039.json', 'dialogues_042.json', 'dialogues_115.json',
                    'dialogues_108.json', 'dialogues_009.json', 'dialogues_038.json', 'dialogues_119.json',
                    'dialogues_124.json', 'dialogues_125.json', 'dialogues_093.json', 'dialogues_005.json',
                    'dialogues_105.json', 'dialogues_056.json', 'dialogues_097.json', 'dialogues_052.json',
                    'dialogues_008.json', 'dialogues_080.json', 'dialogues_101.json', 'dialogues_102.json',
                    'dialogues_057.json', 'dialogues_015.json', 'dialogues_047.json', 'dialogues_043.json',
                    'dialogues_081.json', 'dialogues_028.json', 'dialogues_034.json', 'dialogues_091.json',
                    'dialogues_117.json', 'dialogues_064.json', 'dialogues_082.json', 'dialogues_051.json',
                    'dialogues_109.json', 'dialogues_017.json', 'dialogues_074.json', 'dialogues_012.json',
                    'dialogues_073.json', 'dialogues_035.json', 'dialogues_086.json', 'dialogues_050.json',
                    'dialogues_006.json', 'dialogues_083.json', 'dialogues_024.json', 'dialogues_095.json',
                    'dialogues_079.json', 'dialogues_099.json', 'dialogues_004.json', 'dialogues_021.json',
                    'dialogues_116.json', 'dialogues_020.json', 'dialogues_070.json', 'dialogues_077.json',
                    'dialogues_088.json', 'dialogues_019.json', 'dialogues_053.json']
dev_file_sever = ['dialogues_011.json', 'dialogues_007.json', 'dialogues_018.json',
                  'schema.json',
                  'dialogues_016.json', 'dialogues_003.json', 'dialogues_001.json', 'dialogues_014.json',
                  'dialogues_013.json', 'dialogues_002.json', 'dialogues_010.json', 'dialogues_009.json',
                  'dialogues_005.json', 'dialogues_008.json', 'dialogues_015.json', 'dialogues_017.json',
                  'dialogues_012.json', 'dialogues_006.json', 'dialogues_004.json', 'dialogues_020.json',
                  'dialogues_019.json']
test_file_sever = ['dialogues_011.json', 'dialogues_007.json', 'dialogues_027.json', 'dialogues_026.json',
                   'dialogues_022.json', 'dialogues_018.json', 'schema.json',
                   'dialogues_032.json', 'dialogues_016.json', 'dialogues_003.json', 'dialogues_029.json',
                   'dialogues_001.json', 'dialogues_014.json', 'dialogues_030.json', 'dialogues_013.json',
                   'dialogues_025.json', 'dialogues_031.json', 'dialogues_033.json', 'dialogues_002.json',
                   'dialogues_010.json', 'dialogues_023.json', 'dialogues_009.json', 'dialogues_005.json',
                   'dialogues_008.json', 'dialogues_015.json', 'dialogues_028.json', 'dialogues_034.json',
                   'dialogues_017.json', 'dialogues_012.json', 'dialogues_006.json', 'dialogues_024.json',
                   'dialogues_004.json', 'dialogues_021.json', 'dialogues_020.json', 'dialogues_019.json']

def formalize_schema():
    schema = read_json(args.input_dir+'train/schema.json')
    schema_all = {}
    for domain in schema:
        state_slots = set()
        for intent in domain[ "intents" ]:
            state_slots.update([s.lower().strip() for s in intent[ "required_slots" ]])
            state_slots.update([s.lower().strip() for s in intent[ "optional_slots" ]])

        domain_name = domain['service_name'].lower().strip()
        slot_all = []
        for slot in domain['slots']:
            slot_name = slot['name'].lower().strip()
            if slot['name'] in state_slots:
                slot_all.append(domain_name+'-'+slot_name)
        schema_all[domain_name] = sorted(slot_all)
    return schema_all

def read_json_list(input_dic, file_type):
    per_data = []
    for per_file in file_type:
        if per_file != 'schema.json':
            per_data += read_json(os.path.join(input_dic, per_file))
        else:
            print('ignore schema file')
    return per_data

def main():
    schema = formalize_schema()
    train_data, dev_data, test_data = {}, {},{}
    data_all = {'train':train_data, 'dev':dev_data, 'test':test_data}
    file_all = {'train':train_file_sever, 'dev':dev_file_sever, 'test':test_file_sever}

    for data_style in ['train','dev','test']:
        temp_data = data_all[data_style]

        per_data = read_json_list('data/SGD/'+data_style, file_all[data_style])
        for per_dialog in per_data:
            turns, get_domain, oov_domain = process_dialog(per_dialog, schema)
            if get_domain == '':
                continue

            if not oov_domain:
                domains = get_domain
                per_dialog_normal = {'domains':domains, 'turns':turns}
                if domains in temp_data.keys():
                    temp_data[domains].append(per_dialog_normal)
                else:
                    temp_data[domains] = [per_dialog_normal]
            else:
                print(get_domain)
    for s, v in train_data.items():
        if s in dev_data.keys():
            train_data[s] = train_data[s] + dev_data[s]
        if s in test_data.keys():
            train_data[s] = train_data[s] + test_data[s]

    train_data = dict(sorted(list(train_data.items()), key=lambda x:len(x[1]), reverse=True))

    for lifelong_domain in ['flights_1', 'events_2', 'movies_1-restaurants_1', 'movies_1', 'restaurants_1', 'homes_1',
                            'music_2', 'hotels_2', 'events_1', 'media_1', 'media_1-restaurants_1', 'services_1',
                            'calendar_1-homes_1', 'events_2-restaurants_1', 'events_1-restaurants_1']:
        length_0, length_1 =  int(len(train_data[lifelong_domain])/5*3), int(len(train_data[lifelong_domain])/5*4)
        write_json(train_data[lifelong_domain][:length_0], args.output_dir+'/'+lifelong_domain+'[train.json')
        write_json(train_data[lifelong_domain][length_0:length_1], args.output_dir+'/'+lifelong_domain+'[dev.json')
        write_json(train_data[lifelong_domain][length_1:], args.output_dir+'/'+lifelong_domain+'[test.json')
    return 1

def process_dialog(dialog, schema):
    temp_turns, get_domain, oov_domain = [], [], False
    for per_turn_idx, per_turn in enumerate(dialog['turns']):
        if per_turn_idx % 2 == 0:
            assert per_turn['speaker'] == 'USER'
            user_utterance = per_turn['utterance']
            if per_turn_idx == 0:
                system_utterance = ''
            else:
                system_utterance = dialog['turns'][per_turn_idx-1]['utterance']

            belief_state = []
            for per_frames in per_turn['frames']:
                domain = per_frames['service'].lower().strip()
                get_domain.append(domain)
                for s, v in per_frames['state']['slot_values'].items():
                    s, v = s.lower().strip(), v[0].lower().strip()
                    if v == 'none':
                        continue
                    if domain not in schema.keys():
                        oov_domain = True
                        continue

                    ds = domain + '-' + s
                    if ds in schema[domain]:
                        belief_state.append([ds, v])
                    else:
                        print('1')

            temp_belief_state = sorted(belief_state, key=lambda x:x[0])

            temp_turns.append({'user_utterance':user_utterance,
                               'system_utterance': system_utterance,
                               'belief_state': temp_belief_state,
                               })
    return temp_turns, '-'.join(sorted(list(set(get_domain)))), oov_domain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='data/SGD/')
    parser.add_argument("--output_dir", type=str, default='data/SGD/lifelong')
    args = parser.parse_args()
    main()