import torch
import sentencepiece as spm
import numpy as np
from selfattention import GPTClassifier
from rlbasedmodel import MaskingDQNAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file="peptide_tokenizer.model")
vocab_size = sp.GetPieceSize()

transformer = GPTClassifier(vocab_size).to(device)
transformer.load_state_dict(torch.load("transformer_checkpoint_200.pt", map_location=device))
transformer.eval()

agent = MaskingDQNAgent(state_size=312, action_size=312)
agent.load("masking_agent_checkpoint_200.h5")

def predict_user_sequence(sequence: str):
    
    tokens = sp.Encode(sequence)
    tokens = tokens[:312]  
    tokens += [0] * (312 - len(tokens))
    state = np.array(tokens)

    mask = agent.act(state)
    masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]

    input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)
    logits = transformer(input_tensor)
    pred_label = torch.argmax(logits, dim=-1).item()

    return {
        "mask": mask,
        "masked_input_tokens": masked_input,
        "predicted_label": pred_label
    }

if __name__ == "__main__":
    while True:
        user_seq = input("Enter peptide sequence (or 'exit' to quit): ").strip()
        if user_seq.lower() == "exit":
            break
        result = predict_user_sequence(user_seq)
        print("Predicted Label:", result["predicted_label"])
        print("Mask applied by agent:", result["mask"])
        print("Masked Input Tokens:", result["masked_input_tokens"])
