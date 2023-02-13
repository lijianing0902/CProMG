import torch
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")




class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  
        self.length_penalty = length_penalty 
        self.num_beams = num_beams 
        self.beams = [] 
        self.worst_score = 1e9 
 
    def __len__(self):
        return len(self.beams)
 
    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty 
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
 
    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

      


 
def beam_search(model,smiVoc,num_beams,batch_size,max_length,topk,example, prop=None):

    batch = example.protein_element_batch
    node_attr = example.protein_atom_feature.float()
    pos = example.protein_pos
    aa_node_attr = example.residue_feature.float()
    aa_pos = example.residue_center_of_mass
    aa_batch = example.residue_amino_acid_batch
    atom_laplacian = example.protein_atom_laplacian
    aa_laplacian = example.protein_aa_laplacian
    
    cur_len = 1
    vocab_size = len(smiVoc)
    sos_token_id = smiVoc.index('&')
    eos_token_id = smiVoc.index('$')
    pad_token_id = smiVoc.index('^')
    beam_scores = torch.zeros((batch_size, num_beams)).to(device) 
    beam_scores[:, 1:] = -1e9 
    beam_scores = beam_scores.view(-1) 
    done = [False for _ in range(batch_size)] 
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty=0.7)
            for _ in range(batch_size)
    ] 
    
    input_ids =  torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long).to(device)

    enc_outputs1, enc_pad_mask1, msa_outputs = model.encoder(node_attr, pos, batch, atom_laplacian)
    enc_outputs2, enc_pad_mask2 = model.encoder2(aa_node_attr, aa_pos, aa_batch, aa_laplacian, enc_pad_mask1, msa_outputs)  
    enc_outputs = torch.cat([enc_outputs1,enc_outputs2],dim=1)
    pad_attn_mask = torch.cat([enc_pad_mask1,enc_pad_mask2],dim=2)

    enc_outputs = enc_outputs.repeat_interleave(num_beams,0)
    pad_attn_mask = pad_attn_mask.repeat_interleave(num_beams,0)


    while cur_len < max_length:
        
        dec_outputs = model.decoder(input_ids, enc_outputs, pad_attn_mask, cur_len, prop)
        dec_logits = model.projection(dec_outputs) 
        next_token_logits = dec_logits[:, -1, :]
        scores = F.log_softmax(next_token_logits, dim=-1) 
        next_scores = scores + beam_scores[:, None].expand_as(scores) 
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        ) 
        next_scores, next_tokens = torch.topk(next_scores, 2*num_beams, dim=1, largest=True, sorted=True)
  
        next_batch_beam = []
 
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  
                continue
            next_sent_beam = [] 
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                beam_id = beam_token_id // vocab_size 
                token_id = beam_token_id % vocab_size 

   
                effective_beam_id = batch_idx * num_beams + beam_id
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
 
                if len(next_sent_beam) == num_beams:
                    break

                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                 ) 

            next_batch_beam.extend(next_sent_beam)

        if all(done):
            break

        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        
        input_ids = input_ids[beam_idx, :]
        enc_outputs = enc_outputs[beam_idx, :]
        pad_attn_mask = pad_attn_mask[beam_idx, :]

        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    output_num_return_sequences_per_batch = topk
    output_batch_size = output_num_return_sequences_per_batch * batch_size
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        decoded = torch.stack(best).type(torch.long)

        
    return decoded


