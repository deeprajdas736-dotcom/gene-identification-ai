import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# 1. Define the 'Brain' structure (Must match your training)
class GeneDetector(nn.Module):
    def __init__(self):
        super(GeneDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc(x)
        return x

def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    return np.array([mapping.get(base.upper(), [0,0,0,0]) for base in sequence])

# 2. Setup the Web Interface
st.title("🧬 Gene Identification AI Portal")
st.write("MSc Biotechnology Dissertation - NIT Agartala")

user_dna = st.text_area("Paste DNA Sequence:", "ATGCGTACGTAGCTAGCTAGCTAGCTAGC")

if st.button("Run AI Analysis"):
    # Load your trained model
    model = GeneDetector()
    model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Analyze
    input_tensor = torch.tensor(one_hot_encode(user_dna)).float().T.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence = probs[0][1].item() * 100
        prediction = torch.argmax(output, dim=1).item()

    # Show Results
    status = "GENE DETECTED" if prediction == 1 else "NO GENE FOUND"
    st.success(f"Result: {status} | Confidence: {confidence:.2f}%")

    # Download Button
    df = pd.DataFrame([{"Sequence": user_dna, "Result": status, "Confidence": f"{confidence:.2f}%"}])
    st.download_button("📥 Download Results (CSV)", df.to_csv(index=False), "result.csv", "text/csv")