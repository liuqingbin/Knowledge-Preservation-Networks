import os
from utils.basic_model import utt2tensor, get_mask_metric
from utils.basic_func import read_json
import random
from tqdm import tqdm
import torch

DOMAINS = ['flights_1', 'restaurants_1', 'movies_1-restaurants_1', 'movies_1', 'media_1-restaurants_1', 'media_1', 'music_2',
 'events_2-restaurants_1', 'events_2', 'events_1-restaurants_1', 'events_1', 'calendar_1-homes_1', 'homes_1', 'hotels_2', 'services_1']

gate_dict = {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3}
reverse_gate_dict = {0:'delete', 1:'update', 2:'dontcare', 3:'carryover'}

flatten = lambda x: [i for s in x for i in s]

def get_schema(par, domains):
    schema = read_json(par.raw_data_path + '/schema.json')
    schema_all = {}
    for domain in schema:
        domain_name = domain['service_name'].lower().strip()
        if domain_name not in domains:
            continue

        state_slots = set()
        for intent in domain[ "intents" ]:
            state_slots.update([s.lower().strip() for s in intent[ "required_slots" ]])
            state_slots.update([s.lower().strip() for s in intent[ "optional_slots" ]])

        slot_all = []
        for slot in domain['slots']:
            slot_name = slot['name'].lower().strip()
            if slot['name'] in state_slots:
                slot_all.append(domain_name+'-'+slot_name)
        schema_all[domain_name] = sorted(slot_all)
    return schema_all

def prepare_dataset(par, data, domain, schema, tokenizer):
    data_process = []
    for per_dialog in data:

        dialogue_history, dialog_turns, last_belief_state, last_turn_utterance = [], [], [], ''
        for turn in per_dialog['turns']:
            turn_utterance = (turn[ "system_utterance" ] + ' ; ' + turn[ "user_utterance" ]).lower().strip()
            dialogue_history.append(last_turn_utterance)
            last_turn_utterance = turn_utterance

            y_operation, y_generation = label_num(turn['belief_state'], last_belief_state, schema, tokenizer, per_dialog['domains'])

            dialog_turns.append({
                'dialogue_history': ' '.join(dialogue_history[-par.dialog_turn_num:]),
                'turn_utterance':turn_utterance,
                'last_belief_state': last_belief_state,
                'belief_state':turn['belief_state'],
                'y_operation': y_operation,
                'y_generation': y_generation,
                'domains': per_dialog['domains'],
            })
            last_belief_state = turn['belief_state']
        data_process.append(dialog_turns)

    return data_process

def label_num(belief_state, last_belief_state, schema, tokenizer, domains):

    slots_all = []
    for d in domains.split('-'):
        slots_all += schema[d]
    y_operation = ['carryover'] * len(slots_all)
    y_generate = []

    for sv in belief_state:
        idx = slots_all.index(sv[0])
        if sv in last_belief_state:
            y_operation[idx] = 'carryover'
        else:
            if sv[1] == 'dontcare':
                y_operation[idx] = 'dontcare'
            else:
                y_operation[idx] = 'update'
                y_generate.append([ tokenizer.tokenize(sv[1]) + ['[unused2]'], idx])

    temp_belief_state = dict(belief_state)
    for sv in last_belief_state:
        temp_v = temp_belief_state.get(sv[0])
        if temp_v is None:
            idx = slots_all.index(sv[0])
            y_operation[idx] = 'delete'

    if len(y_generate) > 0:
        y_generate = sorted(y_generate, key=lambda lst: lst[1])
        y_generate, _ = [list(e) for e in list(zip(*y_generate))]

    return y_operation, y_generate

def data_tokenizer_loader(par, data, tokenizer, schema, shuffle, is_train):
    for per_dialog in data:
        slots_all = [ ]
        for d in per_dialog[0]['domains'].split('-'):
            slots_all += schema[d]

        for per_turn in per_dialog:
            last_dialogue_state_dict = dict(per_turn['last_belief_state'])
            belief_state_token = []
            for s in slots_all:
                belief_state_token.append('[unused0]')
                temp_ = s.split('-')
                temp_d, temp_s = temp_[0].split('_'), temp_[1].split('_')
                temp_s = temp_d + temp_s

                v = last_dialogue_state_dict.get(s)
                if v is not None:
                    temp_s.extend(['-', v])
                    temp_s = tokenizer.tokenize(' '.join(temp_s))
                else:
                    temp_s = tokenizer.tokenize(' '.join(temp_s))
                    temp_s.extend([ '-', '[unused1]'])
                belief_state_token.extend(temp_s)

            dialog_history_idx = tokenizer.tokenize(per_turn['dialogue_history'])
            turn_utterance_idx = tokenizer.tokenize(per_turn['turn_utterance'])

            avail_length_1 = par.max_seq_length - len(belief_state_token) - 3
            avail_length = avail_length_1 - len(turn_utterance_idx)

            if len(dialog_history_idx) > avail_length:
                dialog_history_idx = dialog_history_idx[len(dialog_history_idx) - avail_length:]

            if len(dialog_history_idx) == 0 and len(turn_utterance_idx) > avail_length_1:
                turn_utterance_idx = turn_utterance_idx[ len(turn_utterance_idx) - avail_length_1: ]

            tokens = ["[CLS]"] + dialog_history_idx + ["[SEP]"] + turn_utterance_idx + ["[SEP]"]
            segment = [ 0 ] * (len(dialog_history_idx)+2) + [ 1 ] * (len(turn_utterance_idx)+1)
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            per_turn['token_idx'] = token_idx
            per_turn['token_segment_idx'] = segment

            per_turn['last_belief_state_idx'] = tokenizer.convert_tokens_to_ids(belief_state_token)
            per_turn['bs_segment_idx'] = [1]*len(belief_state_token)

            slot_position = []
            for token_idx, token in enumerate(belief_state_token):
                if token == '[unused0]':
                    slot_position.append(token_idx+len(per_turn['token_idx']))
            per_turn['slot_position'] = slot_position

            per_turn['y_operation_num'] = [gate_dict[ope] for ope in per_turn['y_operation']]
            per_turn['y_generation_num'] = [tokenizer.convert_tokens_to_ids(gen) for gen in per_turn['y_generation']]

    data_loader = Dataset_class(par, data, schema, tokenizer, shuffle, is_train)
    return data_loader

class Dataset_class(object):
    def __init__(self, par, data_in, schema, tokenizer, shuffle, is_train):
        super().__init__()
        self.data_in = data_in
        self.par = par
        self.schema = schema
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.shuffle = shuffle
        if is_train:
            self.batch_size = par.batch_size
        else:
            self.batch_size = par.test_batch_size

    def _transpose_batch(self, batch):
        batch_turn = []
        for turn_idx in range(len(self.data_in[batch[0][0]])):
            per_turn_batch = [self.data_in[sample[0]][turn_idx] for sample in batch]
            batch_turn.append(per_turn_batch)
        return batch_turn

    def fill_belief_state(self, batch_turn, predict_last_belief_state, device, training):
        input_id, segment_id, position_idx = [ ], [ ], [ ]
        target_operation, target_generation = [ ], [ ]

        for sample_idx, sample in enumerate(batch_turn):

            if self.par.truth_belief_state & training:
                belief_state_token_idx = sample['last_belief_state_idx']
                position_idx.append(sample['slot_position'])
            else:
                if predict_last_belief_state is None:
                    current_belief_state = {}
                else:
                    current_belief_state = dict(predict_last_belief_state[sample_idx])

                slots_all = [ ]
                for d in sample['domains'].split('-'):
                    slots_all += self.schema[d]

                belief_state_token = [ ]
                for s in slots_all:
                    belief_state_token.append('[unused0]')
                    temp_ = s.split('-')
                    temp_d, temp_s = temp_[0].split('_'), temp_[1].split('_')
                    temp_s = temp_d + temp_s

                    v = current_belief_state.get(s)
                    if v is not None:
                        temp_s.extend([ '-', v ])
                        temp_s = self.tokenizer.tokenize(' '.join(temp_s))
                    else:
                        temp_s = self.tokenizer.tokenize(' '.join(temp_s))
                        temp_s.extend([ '-', '[unused1]' ])
                    belief_state_token.extend(temp_s)
                belief_state_token_idx = self.tokenizer.convert_tokens_to_ids(belief_state_token)

                slot_position = [ ]
                for token_idx, token in enumerate(belief_state_token):
                    if token == '[unused0]':
                        slot_position.append(token_idx + len(sample['token_idx']))
                position_idx.append(slot_position)

            per_input_id = sample['token_idx'] + belief_state_token_idx
            per_segment_id = sample['token_segment_idx'] + [1] * len(belief_state_token_idx)

            if len(per_segment_id) > 512:
                print('over length')
                over_length = len(per_segment_id) - 512
                per_position_all = []
                for per_position in position_idx[-1]:
                    if per_position-over_length >= 0:
                        per_position_all.append(per_position-over_length)
                    else:
                        per_position_all.append(0)
                del position_idx[-1]
                position_idx.append(per_position_all)
                per_input_id = per_input_id[-512:]
                per_segment_id = per_segment_id[-512:]

            input_id.append(per_input_id)
            segment_id.append(per_segment_id)

            target_operation.append(sample['y_operation_num'])
            target_generation.append(sample['y_generation_num'])

        position_idx = torch.tensor(position_idx, dtype=torch.long)
        target_operation = torch.tensor(target_operation, dtype=torch.long)

        max_update = max([len(b) for b in target_generation])
        temp_generation = flatten(target_generation)
        if len(temp_generation) != 0:
            max_value = max([len(b) for b in temp_generation])
        else:
            max_value = 0

        for bid, b in enumerate(target_generation):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            target_generation[bid] = b + [[0] * max_value] * (max_update - n_update)
        target_generation = torch.tensor(target_generation, dtype=torch.long)

        input_id, input_length = utt2tensor(input_id, self.pad_idx, is_float=False)
        segment_id, _ = utt2tensor(segment_id, 0, is_float=False)
        input_mask = get_mask_metric(input_length)

        if max_value > self.par.max_r_len:
            print('cased', max_value, self.par.max_r_len)
        return [input_id.to(device), segment_id.to(device), input_mask.to(device), position_idx.to(device),
                target_operation.to(device), target_generation.to(device)], max_update, max_value

    def mini_batch_iterator(self):
        all_samples = {}
        for dial_idx, dial in enumerate(self.data_in):
            if dial[0]['domains'] not in all_samples.keys():
                all_samples[dial[0]['domains']] = [[dial_idx, len(dial)]]
            else:
                all_samples[dial[0]['domains']].append([dial_idx, len(dial)])

        if self.shuffle:
            for s in all_samples.keys():
                random.shuffle(all_samples[s])

        all_batches_domain = []
        for s in all_samples.keys():
            all_batches, batch = [], {}
            for dial in all_samples[s]:
                if dial[1] not in batch.keys():
                    batch[dial[1]] = []
                batch[dial[1]].append(dial)
                for bk, bv in batch.items():
                    if len(bv) == self.batch_size:
                        all_batches.append(bv)
                        batch[bk] = []
            for bk, bv in batch.items():
                if len(bv) != 0:
                    all_batches.append(bv)
            all_batches_domain = all_batches_domain + all_batches

        for batch in tqdm(all_batches_domain, total=len(all_batches_domain)):
            yield self._transpose_batch(batch)


def increment_dataset(par, domains, data_type):
    data = []
    for per_domain in domains:
        data += read_json(os.path.join(par.data_path, per_domain+'['+data_type+'.json'))
    return data

def decode_belief_state(par, predicts, tokenizer, schema, predict_belief_state, samples):

    belief_state = []
    for b_idx, b_gate in enumerate(predicts[0]):
        if predict_belief_state is not None:
            last_belief_state = dict(predict_belief_state[b_idx])
        else:
            last_belief_state = {}

        predict_operations = [reverse_gate_dict[ a ] for a in b_gate]

        slots_all = [ ]
        for d in samples[b_idx]['domains'].split('-'):
            slots_all += schema[d]

        gid = 0
        for st, op in zip(slots_all, predict_operations):
            if op == 'dontcare':
                last_belief_state[st] = 'dontcare'
            elif (op == 'delete') & (last_belief_state.get(st) is not None):
                last_belief_state.pop(st)
            elif op == 'update':
                if predicts[1] is not None:
                    value = [ ]
                    for v_idx in predicts[1][b_idx][gid]:
                        temp_token = tokenizer.convert_ids_to_tokens(v_idx)
                        if temp_token == '[unused2]':
                            break
                        value.append(temp_token)
                    gid += 1
                    value = ' '.join(value).replace(' ##', '')
                    value = value.replace(' : ', ':').replace('##', '')
                    last_belief_state[st] = value
        belief_state.append(dict_to_list(last_belief_state))

    return belief_state

def dict_to_list(input_dict):
    output_list = []
    for s, v in input_dict.items():
        output_list.append([s, v])
    return output_list

def predicts_to_list(predicts):
    if predicts[1].shape[1] == 0:
        predicts_list = [torch.argmax(predicts[0], dim=2).tolist(), None]
    else:
        predicts_list = [torch.argmax(predicts[0], dim=2).tolist(), torch.argmax(predicts[1], dim=3).tolist()]
    return predicts_list