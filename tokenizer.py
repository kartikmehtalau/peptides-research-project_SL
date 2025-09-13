import sentencepiece as spm
import os
import re

amp_tr = r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.tr.fa"
amp_eval = r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.eval.fa"
amp_te = r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.te.fa"
decoy_tr = r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.tr.fa"
decoy_eval = r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.eval.fa"
decoy_te = r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.te.fa"

text_file_for_spm = "peptide_sequences_for_tokenizer.txt"
vocab_size = 112971

def read_fasta(filepath):
    sequences = []
    with open(filepath, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

amp_sequences = (
    read_fasta(amp_tr) + read_fasta(amp_eval) + read_fasta(amp_te)
)
decoy_sequences = (
    read_fasta(decoy_tr) + read_fasta(decoy_eval) + read_fasta(decoy_te)
)

print(f"AMP sequences: {len(amp_sequences)}")
print(f"Non-AMP (decoy) sequences: {len(decoy_sequences)}")

all_sequences = amp_sequences + decoy_sequences

with open(text_file_for_spm, "w", encoding="utf-8") as f:
    f.write("\n".join(all_sequences))

print(f"Saved {len(all_sequences)} peptide sequences to {text_file_for_spm}")

spm.SentencePieceTrainer.Train(
    input=text_file_for_spm,
    model_prefix="peptide_tokenizer",
    model_type="bpe",
    vocab_size=vocab_size,
    self_test_sample_size=0,
    input_format="text",
    character_coverage=1.0,
    num_threads=os.cpu_count(),
    split_digits=True,
    allow_whitespace_only_pieces=False,
    byte_fallback=True,
    unk_surface=r"\342\201\207",
    normalization_rule_name="identity",
)

print("✅ Peptide tokenizer training completed!")
