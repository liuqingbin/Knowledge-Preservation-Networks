import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from utils.basic_model import orth_gru, Attention_Vector
import random
import torch.nn.functional as F
import math
gate_dict = {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3}

class Bert_DST(BertPreTrainedModel):
    def __init__(self, par, config, pad_idx):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(config, par)
        self.decoder = Decoder(par, config, self.encoder.bert.embeddings.word_embeddings.weight, pad_idx)

    def forward(self, data, schema, max_length, is_update_ewc):

        enc_outputs = self.encoder(data, max_length[0], is_update_ewc)
        gen_scores, hidden_state_all = self.decoder(data, enc_outputs, schema, max_length, is_update_ewc)

        return enc_outputs[2], gen_scores, [enc_outputs[1], enc_outputs[4], hidden_state_all]


class Encoder(nn.Module):
    def __init__(self, config, par):
        super(Encoder, self).__init__()
        self.bert = BertModel(config).from_pretrained(par.bert_base_uncased_path)
        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, 4)
        self.config = config

    def forward(self, data, max_update, is_update_ewc):
        sequence_output, pooled_output = self.bert(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2])
        slot_pos = data[3].unsqueeze(2).repeat(1, 1, sequence_output.shape[-1])
        slot_output = torch.gather(sequence_output, 1, slot_pos)
        state_scores = self.action_cls(self.dropout(slot_output))

        if self.training | is_update_ewc:
            op_idx = data[4]
        else:
            op_idx = torch.argmax(state_scores, dim=2)

        if self.training | is_update_ewc:
            max_update = max_update
        else:
            max_update = op_idx.eq(gate_dict['update']).sum(-1).max().item()

        gathered = []
        for b, a in zip(slot_output, op_idx.eq(gate_dict['update'])):  # update
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.config.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.config.hidden_size).cuda()
                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.config.hidden_size).cuda()
            gathered.append(v)
        decoder_inputs = torch.cat(gathered)

        return sequence_output, pooled_output, state_scores, decoder_inputs, slot_output


class Decoder(nn.Module):
    def __init__(self, par, config, bert_model_embedding_weights, pad_idx):
        super(Decoder, self).__init__()
        self.par, self.vocab_size = par, config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx= pad_idx)
        self.embedding.weight = bert_model_embedding_weights

        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        orth_gru(self.gru)

        self.generation_attention = Attention_Vector(config.hidden_size, dot=True)
        self.generation_gate_w = nn.Linear(config.hidden_size * 3, 1)

        self.hidden_size = config.hidden_size

    def forward(self, data, enc_outputs, schema, max_length, is_update_ewc):

        batch_size, num_update = data[0].shape[0], enc_outputs[3].shape[1]
        max_v_length = max_length[1] if self.training | is_update_ewc else self.par.max_r_len
        all_point_outputs = torch.zeros(num_update, batch_size, max_v_length, self.vocab_size).cuda()
        hidden_state = torch.zeros(num_update, batch_size, max_v_length, self.hidden_size).cuda()
        teacher_force = random.random() < self.par.teacher_force

        for per_update in range(num_update):
            decoder_input = enc_outputs[3][:, per_update].unsqueeze(1)
            hidden = enc_outputs[1].unsqueeze(0)

            for per_value_step in range(max_v_length):
                self.gru.flatten_parameters()
                _, hidden = self.gru(decoder_input, hidden)
                hidden_state[per_update, :, per_value_step, :] = hidden.squeeze(0)

                p_vocab_logits = hidden.squeeze(0).matmul(self.embedding.weight.transpose(1, 0))
                p_vocab = F.softmax(p_vocab_logits, dim=1)  # [B, V]

                logits, prob, context_vec = self.generation_attention(hidden.squeeze(0), enc_outputs[0], (data[2]-1) * (math.pow(2, 32)))
                p_gen_vec = torch.cat([hidden.squeeze(0), context_vec, decoder_input.squeeze(1) ], dim=1)  # [B, 3*H]
                vocab_pointer_switch = torch.sigmoid(self.generation_gate_w(p_gen_vec))  # 从vocab中生成的概率值p_gen
                p_context_ptr = torch.zeros(p_vocab.size()).cuda()  # [B, V]
                p_context_ptr.scatter_add_(1, data[ 0 ], prob)  # 将context范围内copy概率分布映射到词表范围  [B, V]

                final_p_vocab = (1 - vocab_pointer_switch) * p_context_ptr + vocab_pointer_switch * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)  # 选取概率值最高的词作为当前t时刻的生成词  [B]
                if teacher_force & self.training:
                    decoder_input = self.embedding(data[5][:, per_update, per_value_step]).cuda().unsqueeze(1)  # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word).cuda().unsqueeze(1)

                all_point_outputs[per_update, :, per_value_step, :] = final_p_vocab

        return all_point_outputs.transpose(0, 1), hidden_state.transpose(0, 1)
