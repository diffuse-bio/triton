from typing import List
from typing import Tuple
import numpy.typing as npt
import itertools as it

import modal
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pickle as pkl

model_volume = modal.Volume.persisted("esm-model-cache-vol-new")
torch_image = modal.Image.from_registry(
    "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
).pip_install("scipy", "matplotlib")


stub = modal.Stub("ESM2-binder-eval")

# IMPORTANT NOTE - uses AI00 - for shorter sequences (less than 750 AA) please change to T4. 
@stub.function(
    gpu="A100",
    image=torch_image,
    volumes={
        "/root/.cache": model_volume,
    },
)
def get_esm_target_bind_prob_score(sequences: List[str], batch_size :int = 4):
    import numpy as np
    import scipy.special as sp
    import torch
    

    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model.eval()

    batch_converter = alphabet.get_batch_converter()


    sequence_batches: List[List[str]] = [
    sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)
    ]

    sum_log_probs_score: List[float] = []
    inter_contact_regions: List[npt.NDArray[Any]] = []

    for seq_batch in sequence_batches:
        
        data = [("linked_seq", seq[0]) for seq in seq_batch]
       
        separator_idxes = np.array([seq[1] for seq in seq_batch])

        #batch_labels, batch_strs, batch_tokens = batch_converter(data) # to do: use meaningful labels and names for peptide for keeping track of results
        _, _, batch_tokens = batch_converter(data)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True) # returns both contacts and attention maps
            #results = model.predict_contacts(batch_tokens)

        # if using predict_contacts option disable / change following assertion. 
        # predict_contact returns only contacts (trimmed to actual size, 0th and last token appened by tokenizer is removed)
        
        assert set(results.keys()) == set(
            [
                "logits", "representations",
                "attentions", "contacts"
            ]
        )


        # each sequence in the batch has different protein / peptide lengths, we have to process each sequene individually

        for b_idx in range(batch_size):

            target_prot_length, linker_length, peptide_length = separator_idxes[b_idx, 0], separator_idxes[b_idx, 1], separator_idxes[b_idx, 2]

            inter_contact_region_top_right = torch.zeros((target_prot_length, peptide_length), dtype=torch.float32) # do we need to add device in modal
            #inter_contact_region_bot_left = torch.zeros((peptide_length, target_prot_length), dtype=torch.float32) # symmetry assumed.

            #save as an image - verify the contact map. Need the right naming label to return the contact images as an array
            pep_start_idx = target_prot_length + linker_length  # contact maps are tripped. (does not include start/end tokens)
            inter_contact_region_top_right[ : , : ] = results["contacts"][b_idx, : target_prot_length, pep_start_idx : pep_start_idx+peptide_length]

            # Modal execution model would help - converting torch tensor to nd array
            inter_contact_regions.append(inter_contact_region_top_right.numpy())
            # we need to figure out the intra-residue contact in the peptide. Remove the interal contact peptide from the score
            
            # the logits for all indices in the above region indicate inter-contact between protein and peptide.
            seq_logits = torch.zeros((peptide_length, 33), dtype=torch.float32)

            # tokenizer appends start / end tokens - account for the same in copying the logits for the peptide.
            pep_start_idx = target_prot_length + linker_length + 1

            # note - for some sequences, batch size > sequence size. peptide end is not end of sequence for such cases.
            seq_logits[ : , : ] = results["logits"][b_idx, pep_start_idx : pep_start_idx + peptide_length , : ]
            
            # should we computer log probs first and then extract the probs for the peptide sequence
            log_prob = torch.nn.functional.log_softmax(seq_logits, dim=1)

            av_log_prob = sp.logsumexp (log_prob) - np.log(peptide_length)
 
            sum_log_probs_score.append(av_log_prob)

    return inter_contact_regions
    #return sum_log_probs_score


def compute_esm_bind_aff_score_parallel(
    sequences: List[Tuple [str, List[int]]], num_workers: int = 2, batch_size_per_worker: int = 2
):
    num_per_worker = int(np.ceil(len(sequences) / num_workers))
    assert num_per_worker * num_workers >= len(sequences)
   
    worker_split = [
        sequences[i : i + num_per_worker]
        for i in range(0, len(sequences), num_per_worker)
    ]
    args = [(worker_seqs, batch_size_per_worker) for worker_seqs in worker_split]
    
    return list(get_esm_target_bind_prob_score.starmap(args))
    #return np.concatenate(list(get_esm_target_bind_prob_score.starmap(args)))

## following not yet integrated.

#TO DO: validate user supplied link_residue. ()
# reduce 3 input options to two (remove link_default_flag)
# edge case - what is an user wants to use G4S 10 times (not the default 6)

def link_prot_binder (tgt_protein: str, binder_peptide : str,
                      link_default_flag = "G4S",
                      link_residue = "GGGS", link_repeat=6 ):

  # default or custom
  if link_default_flag == "G4S":
    link_str = LINKER_G4S * NUM_G4S_REPEAT
  elif link_default_flag == "G4S":
    link_str = LINKER_POLY_G * NUM_POLY_GLY_REPEAT
  elif link_default_flag.lower() == 'custom':
    try:
      link_str = link_residue * link_repeat
    except:
      print ('Link residue or number of repeats not specified. ')
  else:
    print ("Error - link residue options not specified ")
    return

  # keep track of various indices 
  # to DO: append lengths to the returned sequence (as a tuple)
  tgt_prot_start_idx = 0 # assuming tgt_prot is first - ( binder first to be implemented)
  tgt_prot_end_idx = len(tgt_protein) - 1
  binder_start_idx = len (tgt_protein) + len(link_str)
  binder_end_idx = binder_start_idx + len(binder_peptide) # may not need to calcualte explicitly

  return tgt_protein + link_str + binder_peptide


@stub.local_entrypoint()
def main():
    tar_prot_1 = "MNQNLLVTKRDGSTERINLDKIHRVLDWAAEGLHNVSISQVELRSHIQFYDGIKTSDIHET" \
        "IIKAAADLISRDAPDYQYLAARLAIFHLRKKAYGQFEPPALYDHVVKMVEMGKYDNHLLEDY" \
        "TEEEFKQMDTFIDHDRDMTFSYAAVKQLEGKYLVQNRVTGEIYESAQFLYILVAACLFSNYPRET" \
        "RLQYVKRFYDAVSTFKISLPTPIMSGVRTPTRQFSSCVLIECGDSLDSINATSSAIVKYVSQRAGI" \
        "GINAGRIRALGSPIRGGEAFHTGCIPFYKHFQTAVKSCSQGGVRGGAATLFYPMWHLEVESLLVLKN" \
        "NRGVEGNRVRHMDYGVQINKLMYTRLLKGEDITLFSPSDVPGLYDAFFADQEEFERLYTKYEKDDSIRK" \
        "QRVKAVELFSLMMQERASTGRIYIQNVDHCNTHSPFDPAIAPVRQSNLCLEIALPTKPLNDVNDENGEIAL" \
        "CTLSAFNLGAINNLDELEELAILAVRALDALLDYQDYPIPAAKRGAMGRRTLGIGVINFAYYLAKHGKRYS" \
        "DGSANNLTHKTFEAIQYYLLKASNELAKEQGACPWFNETTYAKGILPIDTYKKDLDTIANEPLHYDWEALRE" \
        "SIKTHGLRNSTLSALMPSETSSQISNATNGIEPPRGYVSIKASKDGILRQVVPDYEHLHDAYELLWEMPGNDG" \
        "YLQLVGIMQKFIDQSISANTNYDPSRFPSGKVPMQQLLKDLLTAYKFGVKTLYYQNTRDGAEDAQDDLVPSIQDDGCESGACKI"

    peptide_1 = "MQTVIFGRSGCPYCVRAKDLAEKLSNERDDFQYQYVDIRAEGITKEDLQQKAGKPVETVPQIFVDQQHIGGYTDFAAWVKENLDA"

    tar_prot_2 = "MNAETCVSYCESPAAAMDAYYSPVSQSREGSSPFRGFPGGDKFGTTFLSAGAKGQGFG" \
            "DAKSRARYGAGQQDLAAPLESSSGARGSFNKFQPQPPTPQPPPAPPAPPAHLYLQRGACKTP" \
            "PDGSLKLQEGSGGHNAALQVPCYAKESNLGEPELPPDSEPVGMDNSYLSVKETGAKGPQDRASA" \
            "EIPSPLEKTDSESNKGKKRRNRTTFTSYQLEELEKVFQKTHYPDVYAREQLAMRTDLTEARVQV" \
            "WFQNRRAKWRKRERFGQMQQVRTHFSTAYELPLLTRAENYAQIQNPSWIGNNGAASPVPACVVPC" \
            "DPVPACMSPHAHPPGSGASSVSDFLSVSGAGSHVGQTHMGSLFGAAGISPGLNGYEMNGEPDRKTSSIAALRMKAKEHSAAISWAT"

    peptide_2 = "MALTSSSQAETWSLHPRASTASLPLGPQEQEAGGSPGASGGLPLEKVKRPMNAFMVWSSVQRRQ" \
            "MAQQNPKMHNSEISKRLGAQWKLLGDEEKRPFVEEAKRLRARHLRDYPDYKYRPRRKSKNSSTGS" \
            "VPFSQEGGGLACGGSHWGPGYTTTQGSRGFGYQPPNYSTAYLPGSYTSSHCRPEAPLPCTFPQSDP" \
            "RLQGELRPSFSPYLSPDSSTPYNTSLAGAPMPVTHL"

    linker = "GGGGSGGGGSGGGGSGGGGSGGGGSGGGGS"


    test_seq_1 = tar_prot_1 + linker + peptide_1
    test_seq_2 = tar_prot_2 + linker + peptide_2

    t_len_1 = len(tar_prot_1)
    p_len_1 = len(peptide_1)
    t_len_2 = len(tar_prot_2)
    p_len_2 = len(peptide_2)
    linker_len = len(linker)

    sequences = [ (test_seq_1, [t_len_1, linker_len, p_len_1]), (test_seq_2, [t_len_2, linker_len, p_len_2])]

    seq_lengths = []

    test_sequences: List[Tuple [str, List[int]]] = list(
        it.islice(it.chain.from_iterable(it.repeat(sequences)), 4)
    )


    contact_regions  = compute_esm_bind_aff_score_parallel(test_sequences)



    np.save("pep1", contact_regions[0][0])
    np.save("pep2", contact_regions[0][1])
    
    fig, ax = plt.subplots()
    ax1 = plt.subplot(221)
    fig.suptitle ("Contact regions between Target Protein and Peptide")
    im1 = ax1.imshow(contact_regions[0][0], cmap='hot')
    plt.colorbar(im1, ax=ax1)
    ax2 = plt.subplot(222)
    im2 = ax2.imshow(contact_regions[0][1], cmap='hot')
    plt.colorbar(im2, ax=ax2)

    plt.show()
    

