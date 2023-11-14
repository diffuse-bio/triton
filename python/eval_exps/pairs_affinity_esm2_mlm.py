# ESM2 MLM for binidng affinitiy between two protein sequences

# implementation of the baseline method outlined in DiffPALM paper
# (ref:  https://www.biorxiv.org/content/10.1101/2023.11.03.565471v1.full.pdf )

# SINGLE Worker code. 
# TO DO: multi-worker (parallel) version - dependency - MODAL execution / caching understanding

from typing import List, Tuple
import numpy.typing as npt

import modal
import numpy as np
from scipy.optimize import linear_sum_assignment


model_volume = modal.Volume.persisted("esm-model-cache-vol-new")
torch_image = modal.Image.from_registry(
    "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
).pip_install("scipy", "transformers")

stub = modal.Stub("ESM2-two-sequences-MLM-loss-eval")

# IMPORTANT NOTE - depending on the model used may need to switch to A100
# esm2_t6_8M_UR50D works well on T4 upto 1022 sequence length (total)
@stub.function(
    gpu="T4",
    image=torch_image,
    volumes={
        "/root/.cache": model_volume,
    },
)
# Function to compute MLM loss for a batch of protein pairs
def mask_and_compute_mlm_loss_batch(batch_prot_pairs: List[Tuple[List[int], List[str]]], tot_num_seqs, mask_prob: float = 0.15, num_iter: int = 10):
  import numpy as np
  from transformers import AutoTokenizer, EsmForMaskedLM
  import torch


  # following works well on T4 upto 1022 total sequence length 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
  model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

  # experimenting with Modal to see if I need to move to GPU - it is the default. 
  # there is no need to move model / data to GPU
  model.eval()
  model = model.to(device)

  # following will require A100 GPU depending on the lenght of sequences
  # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
  # model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")


  loss_matrix = np.zeros((tot_num_seqs, tot_num_seqs))

 
  for batch_data in batch_prot_pairs:
    # each batch
    batch_avg_losses = []
    num_items_in_batch = len(batch_data[1]) # last batch will have less items
    batch_iteration_losses = []

    for _ in range(num_iter): # number of iterations for calculation MLM loss 

      # tokenize the concatenated protein pairs in the batch
      inputs = tokenizer(batch_data[1], return_tensors="pt", truncation=True, padding=True, max_length=1022)
      # move input tensors to GPU if available (for Modal is this needed)
      inputs = {k: v.to(device) for k, v in inputs.items()}

      # mask token ID
      mask_token_id = tokenizer.mask_token_id

      # input IDs for labels (ground truth before masking)
      labels = inputs["input_ids"].clone()

      # Randomly mask mask_prob (*100 percentage) of the residues for each sequence in the batch
      for idx in range(inputs["input_ids"].shape[0]):
        mask_indices = np.random.choice(inputs["input_ids"].shape[1], size=int(mask_prob * inputs["input_ids"].shape[1]), replace=False)
        inputs["input_ids"][idx, mask_indices] = mask_token_id
        labels[idx, [i for i in range(inputs["input_ids"].shape[1]) if i not in mask_indices]] = -100

      # compute the MLM loss
      outputs = model(**inputs, labels=labels)       
      batch_iteration_losses.append(outputs.loss.item())

    avg_batch_loss = sum(batch_iteration_losses) / num_iter

    # TO DO: FOR MULTIPLE WORKERS 
    # simply return batch indices and avg_batch_loss
    # assemble the matrix from losses / indices provided by different workers
      
    # for single worker - update loss matrix
    i, j = batch_data[0][0], batch_data[0][1] 
    for k in range(num_items_in_batch):
      loss_matrix [i, j+k] = avg_batch_loss
      loss_matrix [j+k, i] = avg_batch_loss
    
    
  return loss_matrix


def create_batched_protein_pairs (all_proteins: List [str], linker_seq: str = '', batch_size: int =4) -> List[List[str]]:
  """
  Input: list of all proteins [p1, p2, .... pn]
  Output: List of tuples. each tuple has two lists 
          list of (i, j) index and the list of paired proteins. [ ([1,2][p1p2, p1p3, ..]), ([2,3], [p2p3, p2p4..], .. ) linker not shown

   The size of nested list with protein pairs is batch_size
   Note: the pair: p1 <linker> p2 is considered same as the pair: p2 <linker> p1 
  """
  batched_prot_pairs = [] 
  for i in range(len(all_proteins)):
    for j in range(i+1, len(all_proteins), batch_size):  
      prot_pairs = [all_proteins[i]+ linker_seq + all_proteins[k] for k in range (j, min(j+batch_size, len(all_proteins)))] 
      batched_prot_pairs.append(([i, j], prot_pairs))

  return batched_prot_pairs


@stub.local_entrypoint()
def main():

  test_proteins = [
    "MEESQSELNIDPPLSQETFSELWNLLPENNVLSSELCPAVDELLLPESVVNWLDEDSDDAPRMPATSAPTAPGPAPSWPLSSSVPSPKTYPGTYGFRLGFLHSGTAKSVTWTYSPLLNKLFCQLAKTCPVQLWVSSPPPPNTCVRAMAIYKKSEFVTEVVRRCPHHERCSDSSDGLAPPQHLIRVEGNLRAKYLDDRNTFRHSVVVPYEPPEVGSDYTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNVLGRNSFEVRVCACPGRDRRTEEENFHKKGEPCPEPPPGSTKRALPPSTSSSPPQKKKPLDGEYFTLQIRGRERYEMFRNLNEALELKDAQSGKEPGGSRAHSSHLKAKKGQSTSRHKKLMFKREGLDSD",
    "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSENRCHLEGGSDQKDLVQELQEEKPSSSHLVSRPSTSSRRRAISETEENSDELSGERQRKRHKSDSISLSFDESLALCVIREICCERSSSSESTGTPSNPDLDAGVSEHSGDWLDQDSVSDQFSVEFEVESLDSEDYSLSEEGQELSDEDDEVYQVTVYQAGESDTDSFEEDPEISLADYWKCTSCNEMNPPLPSHCNRCWALRENWLPEDKGKDKGEISEKAKLENSTQAEEGFDVPDCKKTIVNDSRESCVEENDDKITQASQSQESEDYSQPSTSSSIIYSSQEDVKEFEREETQDKEESVESSLPLNAIEPCVICQGRPKNGCIVHGKTGHLMACFTCAKKLKKRNKPCPVCRQPIQMIVLTYFP",
    "MNRGVPFRHLLLVLQLALLPAATQGKKVVLGKKGDTVELTCTASQKKSIQFHWKNSNQIKILGNQGSFLTKGPSKLNDRADSRRSLWDQGNFPLIIKNLKIEDSDTYICEVEDQKEEVQLLVFGLTANSDTHLLQGQSLTLTLESPPGSSPSVQCRSPRGKNIQGGKTLSVSQLELQDSGTWTCTVLQNQKKVEFKIDIVVLAFQKASSIVYKKEGEQVEFSFPLAFTVEKLTGSGELWWQAERASSSKSWITFDLKNKEVSVKRVTQDPKLQMGKKLPLHLTLPQALPQYAGSGNLTLALEAKTGKLHQEVNLVVMRATQLQKNLTCEVWGPTSPKLMLSLKLENKEAKVSKREKAVWVLNPEAGMWQCLLSDSGQVLLESNIKVLPTWSTPVQPMALIVLGGVAGLLLFIGLGIFFCVRCRHRRRQAERMSQIKRLLSEKKTCQCPHRFQKTCSPI",
    "MRVKEKYQHLWRWGWKWGTMLLGILMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLGNATNTNSSNTNSSSGEMMMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYTLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSANFTDNAKTIIVQLNQSVEINCTRPNNNTRKSIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNATLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQFINMWQEVGKAMYAPPISGQIRCSSNITGLLLTRDGGNNNNGSEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGARSMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNNMTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKIFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL",
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "MATGGRRGAAAAPLLVAVAALLLGAAGHLYPGEVCPGMDIRNNLTRLHELENCSVIEGHLQILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYALVIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNYIVLNKDDNEECGDICPGTAKGKTNCPATVINGQFVERCWTHSHCQKVCPTICKSHGCTAEGLCCHSECLGNCSQPDDPTKCVACRNFYLDGRCVETCPPPYYHFQDWRCVNFSFCQDLHHKCKNSRRQGCHQYVIHNNKCIPECPSGYTMNSSNLLCTPCLGPCPKVCHLLEGEKTIDSVTSAQELRGCTVINGSLIINIRGGNNLAAELEANLGLIEEISGYLKIRRSYALVSLSFFRKLRLIRGETLEIGNYSFYALDNQNLRQLWDWSKHNLTITQGKLFFHYNPKLCLSEIHKMEEVSGTKGRQERNDIALKTNGDQASCENELLKFSYIRTSFDKILLRWEPYWPPDFRDLLGFMLFYKEAPYQNVTEFDGQDACGSNSWTVVDIDPPLRSNDPKSQNHPGWLMRGLKPWTQYAIFVKTLVTFSDERRTYGAKSDIIYVQTDATNPSVPLDPISVSNSSSQIILKWKPPSDPNGNITHYLVFWERQAEDSELFELDYCLKGLKLPSRTWSPPFESEDSQKHNQSEYEDSAGECCSCPKTDSQILKELEESSFRKTFEDYLHNVVFVPRKTSSGTGAEDPRPSRKRRSLGDVGNVTVAVPTVAAFPNTSSTSVPTSPEEHRPFEKVVNKESLVISGLRHFTGYRIELQACNQDTPEERCSVAAYVSARTMPEAKADDIVGPVTHEIFENNVVHLMWQEPKEPNGLIVLYEVSYRRYGDEELHLCVSRKHFALERGCRLRGLSPGNYSVRIRATSLAGNGSWTEPTYFYVTDYLDVPSNIAKIIIGPLIFVFLFSVVIGSIYLFLRKRQPDGPLGPLYASSNPEYLSASDVFPCSVYVPDEWEVSR"
    ]


  test_pairs_list = create_batched_protein_pairs(test_proteins, "", 2)

  score_mat = mask_and_compute_mlm_loss_batch.remote(test_pairs_list, tot_num_seqs=6, mask_prob = 0.15, num_iter = 10)
  np.fill_diagonal(score_mat, np.inf)

  rows, cols = linear_sum_assignment(score_mat)
  optimal_pairs = list(zip(rows, cols))
  print (rows)
  print (cols)
  print ('Pairs of protein sequences with increasing MLM loss')
  print (optimal_pairs)
  
