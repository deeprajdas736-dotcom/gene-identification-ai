import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# 1. Model Architecture
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

# 2. Page Setup
st.set_page_config(page_title="AI Gene Identifier", page_icon="🧬")
st.title("🧬 AI Gene Identifier & Extractor")
st.write("Upload or paste DNA to identify specific gene coordinates.")

user_dna = st.text_area("Paste DNA Sequence:", height=200)
WINDOW_SIZE = 30 # Must match your training size
STRIDE = 5      # Speed of scan

if st.button("Identify Genes"):
    # Load Model
    model = GeneDetector()
    model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
    model.eval()

    findings = []
    
    # 3. Sliding Window Identification Logic
    with st.spinner("Scanning sequence..."):
        for i in range(0, len(user_dna) - WINDOW_SIZE, STRIDE):
            window = user_dna[i : i + WINDOW_SIZE]
            input_tensor = torch.tensor(one_hot_encode(window)).float().T.unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence = probs[0][1].item() * 100
                
            if confidence > 80: # Identification Threshold
                findings.append({
                    "Start": i,
                    "End": i + WINDOW_SIZE,
                    "Sequence": window,
                    "Confidence": f"{confidence:.2f}%"
                })

    # 4. Show Results
    if findings:
        st.success(f"Identified {len(findings)} potential gene regions!")
        df = pd.DataFrame(findings)
        st.dataframe(df) # Shows the table of identified genes
        
        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Identified Genes (CSV)", csv, "identified_genes.csv", "text/csv")
    else:
        st.warning("No genes identified with high confidence. Try a different sequence.")
