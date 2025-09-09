import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import numpy as np
import random
import string
from selfattention import GPTClassifier
from rlbasedmodel import MaskingDQNAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file="peptide_tokenizer.model")
vocab_size = sp.GetPieceSize()

valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_peptide(seq):
    return all(residue in valid_aas for residue in seq.upper())

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

def random_invalid_sequence(length):
    letters = string.ascii_uppercase
    invalid = [l for l in letters if l not in valid_aas]
    return ''.join(random.choices(invalid, k=length))

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

synthetic_negatives = [(random_invalid_sequence(10), 0) for _ in range(1000)]

dataset = [(seq, 1) for seq in amp_sequences] + \
          [(seq, 0) for seq in decoy_sequences] + \
          synthetic_negatives
random.shuffle(dataset)

transformer = GPTClassifier(vocab_size).to(device)
optimizer = optim.Adam(transformer.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
agent = MaskingDQNAgent(state_size=312, action_size=312)

batch_size = 32
checkpoint_interval = 100
epochs = 3

print("🔹 Training Transformer with DQN masking on AMP + non-AMP sequences...")

for epoch in range(epochs):
    total_loss = 0
    correct = 0

    for i, (seq, label) in enumerate(dataset):
        tokens = sp.Encode(seq)[:312]
        tokens += [0] * (312 - len(tokens))
        state = np.array(tokens)

        mask = agent.act(state)

        if not is_valid_peptide(seq):
            mask = np.ones(312) 
            label = 0  

        masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]
        input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)

        transformer.train()
        optimizer.zero_grad()
        logits = transformer(input_tensor)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_label = torch.argmax(logits, dim=-1).item()
        if pred_label == label:
            correct += 1

        pred_prob = torch.softmax(logits, dim=-1)[0, label].item()
        reward = pred_prob - 0.1 * (mask.sum() / len(mask))  
        next_state = state.copy()
        done = True
        agent.remember(state, mask, reward, next_state, done)
        agent.replay(batch_size)

        if (i+1) % checkpoint_interval == 0:
            agent.save(f"masking_agent_checkpoint_{i+1}.h5")
            torch.save(transformer.state_dict(), f"transformer_checkpoint_{i+1}.pt")
            print(f"Checkpoint saved | Step {i+1} | Last Reward: {reward}")

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataset):.4f} | Accuracy: {acc:.4f}")

agent.save("masking_agent_final.h5")
torch.save(transformer.state_dict(), "transformer_final.pt")
print("✅ Joint training complete. Transformer and DQN agent saved.")

while True:
    seq = input("Enter peptide sequence (or 'exit' to quit): ")
    if seq.lower() == 'exit':
        break
    if not is_valid_peptide(seq):
        print("Predicted Label: 0 (invalid peptide)")
        continue

    tokens = sp.Encode(seq)[:312]
    tokens += [0] * (312 - len(tokens))
    state = np.array(tokens)

    mask = agent.act(state)
    masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]
    input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)

    transformer.eval()
    with torch.no_grad():
        pred_label = torch.argmax(transformer(input_tensor), dim=-1).item()

    print(f"Predicted Label: {pred_label}")
    print(f"Mask applied by agent: {mask}")
    print(f"Masked Input Tokens: {masked_input}")
