import argparse
from utils.basic_func import read_json, write_json
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
slot_convert = {'pricerange':'price range', 'leaveat':'leave at', 'arriveby':'arrive by'}
def formalize_schema():
    schema = read_json(args.input_dir+'ontology.json')
    slot_all = []
    for ds_idx, ds in enumerate(schema.keys()):
        domain, slot = ds.split('-')
        if domain not in EXPERIMENT_DOMAINS:
            continue
        slot_all.append(ds)
    return sorted(slot_all)


def main():
    slot_all = formalize_schema()
    train_data, dev_data, test_data = {}, {},{}
    data_all = {'train':train_data, 'dev':dev_data, 'test':test_data}


    for data_style in ['train','dev','test']:
        temp_data = data_all[data_style]

        per_data = read_json(args.input_dir+'split/'+data_style+'_dials.json')
        for per_dialog in per_data:
            turns, get_domain = process_dialog(per_dialog, slot_all)
            if get_domain == '':
                continue
            domains = get_domain
            per_dialog_normal = {'domains':domains, 'turns':turns}
            if domains in temp_data.keys():
                temp_data[domains].append(per_dialog_normal)
            else:
                temp_data[domains] = [per_dialog_normal]

    train_data = dict(sorted(list(train_data.items()), key=lambda x:len(x[1]), reverse=True))

    for lifelong_domain in [ 'restaurant', 'hotel', 'hotel-restaurant', 'train', 'restaurant-train', 'hotel-train',
                             'attraction-restaurant', 'attraction', 'attraction-hotel', 'attraction-train' ]:
        write_json(train_data[lifelong_domain], args.output_dir+'/'+lifelong_domain+'[train.json')
        write_json(dev_data[lifelong_domain], args.output_dir+'/'+lifelong_domain+'[dev.json')
        write_json(test_data[lifelong_domain], args.output_dir+'/'+lifelong_domain+'[test.json')
    return 1

def process_dialog(dialog, slot_all):
    temp_turns, get_domain = [], []
    for per_turn_idx, per_turn in enumerate(dialog['dialogue']):
        user_utterance = per_turn['transcript']
        system_utterance = per_turn['system_transcript']

        belief_state = []
        for sv in per_turn[ 'belief_state' ]:
            assert len(sv['slots']) == 1
            s, v = sv['slots'][0][0].lower().strip(), sv['slots'][0][1].lower().strip()
            if v == 'none':
                continue
            d = s.split('-')[0]
            belief_state.append([s, v])
            get_domain.append(d)
        temp_belief_state = rewrite_slot(belief_state)
        temp_belief_state = fix_general_label_error(temp_belief_state, True, slot_all)

        list_belief_state = []
        for s,v in temp_belief_state.items():
            list_belief_state.append([s, v])
        temp_belief_state = sorted(list_belief_state, key=lambda x:x[0])

        temp_turns.append({'user_utterance':user_utterance,
                           'system_utterance': system_utterance,
                           'belief_state': temp_belief_state,
                           })
    return temp_turns, '-'.join(sorted(list(set(get_domain))))

def rewrite_slot(belief_state):
    temp_belief_state = []
    for sv in belief_state:
        if ('price' in sv[0]) or ('leave' in sv[0])or ('arrive' in sv[0]):
            ds = sv[0].split('-')
            temp_belief_state.append([ds[0]+'-'+slot_convert.get(ds[1], ds[1]), sv[1]])
        else:
            temp_belief_state.append(sv)
    return temp_belief_state

def fix_general_label_error(labels, type, slots):
    label_dict = dict([ (l[ 0 ], l[ 1 ]) for l in labels ]) if type else dict(
        [ (l[ "slots" ][ 0 ][ 0 ], l[ "slots" ][ 0 ][ 1 ]) for l in labels ])

    GENERAL_TYPO = {
        # type
        "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
        "ol": "architecture",
        "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
        "cen": "centre", "east side": "east",
        "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre",
        "in town": "centre", "north part of town": "north",
        "centre of town": "centre", "cb30aq": "none",
        # price
        "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
        # day
        "next friday": "friday", "monda": "monday",
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4", "4 stars": "4", "0 star rarting": "none",
        # others
        "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none",
        "not mentioned": "none",
        '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
    }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[ slot ] in GENERAL_TYPO.keys():
                label_dict[ slot ] = label_dict[ slot ].replace(label_dict[ slot ], GENERAL_TYPO[ label_dict[ slot ] ])

            # miss match slot and value
            if slot == "hotel-type" and label_dict[ slot ] in [ "nigh", "moderate -ly priced", "bed and breakfast",
                                                                "centre", "venetian", "intern", "a cheap -er hotel" ] or \
                    slot == "hotel-internet" and label_dict[ slot ] == "4" or \
                    slot == "hotel-price range" and label_dict[ slot ] == "2" or \
                    slot == "attraction-type" and label_dict[ slot ] in [ "gastropub", "la raza", "galleria", "gallery",
                                                                          "science", "m" ] or \
                    "area" in slot and label_dict[ slot ] in [ "moderate" ] or \
                    "day" in slot and label_dict[ slot ] == "t":
                label_dict[ slot ] = "none"
            elif slot == "hotel-type" and label_dict[ slot ] in [ "hotel with free parking and free wifi", "4",
                                                                  "3 star hotel" ]:
                label_dict[ slot ] = "hotel"
            elif slot == "hotel-star" and label_dict[ slot ] == "3 star hotel":
                label_dict[ slot ] = "3"
            elif "area" in slot:
                if label_dict[ slot ] == "no":
                    label_dict[ slot ] = "north"
                elif label_dict[ slot ] == "we":
                    label_dict[ slot ] = "west"
                elif label_dict[ slot ] == "cent":
                    label_dict[ slot ] = "centre"
            elif "day" in slot:
                if label_dict[ slot ] == "we":
                    label_dict[ slot ] = "wednesday"
                elif label_dict[ slot ] == "no":
                    label_dict[ slot ] = "none"
            elif "price" in slot and label_dict[ slot ] == "ch":
                label_dict[ slot ] = "cheap"
            elif "internet" in slot and label_dict[ slot ] == "free":
                label_dict[ slot ] = "yes"

            # some out-of-define classification slot values
            if slot == "restaurant-area" and label_dict[ slot ] in [ "stansted airport", "cambridge",
                                                                     "silver street" ] or \
                    slot == "attraction-area" and label_dict[ slot ] in [ "norwich", "ely", "museum",
                                                                          "same area as hotel" ]:
                label_dict[ slot ] = "none"

    return label_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='data/MultiWOZ_2.1/')
    parser.add_argument("--data_set", type=str, default='2.1')
    parser.add_argument("--output_dir", type=str, default='data/MultiWOZ_2.1/lifelong')
    args = parser.parse_args()
    main()