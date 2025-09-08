import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import numpy as np
import random
from selfattention import GPTClassifier
from rlbasedmodel import MaskingDQNAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file="peptide_tokenizer.model")
vocab_size = sp.GetPieceSize()

transformer = GPTClassifier(vocab_size).to(device)
optimizer = optim.Adam(transformer.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

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
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.tr.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.eval.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.te.fa")
)
decoy_sequences = (
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.tr.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.eval.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.te.fa")
)

dataset = [(seq, 1) for seq in amp_sequences] + [(seq, 0) for seq in decoy_sequences]
random.shuffle(dataset)

agent = MaskingDQNAgent(state_size=312, action_size=312)
batch_size = 32
checkpoint_interval = 100  

for i, (seq, label) in enumerate(dataset):
    tokens = sp.Encode(seq)
    tokens = tokens[:312]
    tokens += [0] * (312 - len(tokens))
    state = np.array(tokens)

    mask = agent.act(state)
    masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]

    input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)

    transformer.train()
    optimizer.zero_grad()
    logits = transformer(input_tensor)
    loss = criterion(logits, label_tensor)
    loss.backward()
    optimizer.step()

    transformer.eval()
    with torch.no_grad():
        pred_label = torch.argmax(transformer(input_tensor), dim=-1).item()
    reward = 1 if pred_label == label else -1

    next_state = state.copy()
    done = True
    agent.remember(state, mask, reward, next_state, done)
    agent.replay(batch_size)

    if (i + 1) % checkpoint_interval == 0:
        agent_checkpoint = f"masking_agent_checkpoint_{i+1}.h5"
        transformer_checkpoint = f"transformer_checkpoint_{i+1}.pt"
        agent.save(agent_checkpoint)
        torch.save(transformer.state_dict(), transformer_checkpoint)
        print(f"✅ Checkpoint saved: {agent_checkpoint}, {transformer_checkpoint} | Step: {i+1} | Last Reward: {reward}")

agent.save("masking_agent_final.h5")
torch.save(transformer.state_dict(), "transformer_final.pt")
print("✅ Training complete. Final models saved.")
