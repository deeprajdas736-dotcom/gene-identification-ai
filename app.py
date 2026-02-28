import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio.Blast import NCBIWWW, NCBIXML

# 1. AI Model Architecture (Must match your training)
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

# 2. Robust Identification Function
def identify_gene_function(sequence):
    try:
        # Increase timeout to 90 seconds to avoid URLError
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence, timeout=90)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return f"**Identity:** {top_hit.title}\n\n**Length:** {top_hit.length} bp"
        return "No known match found in NCBI database (Potential novel sequence/IDP)."
    except Exception as e:
        return "⚠️ Connection to NCBI timed out. Please wait a moment and click 'Identify' again."

# 3. Streamlit Interface
st.set_page_config(page_title="NIT Agartala Gene AI", page_icon="🧬")
st.title("🧬 AI Gene Identification & Functional Analysis")
st.markdown("Developed for MSc Biotechnology Dissertation - NIT Agartala")

user_dna = st.text_area("Paste DNA Sequence:", "ATGCGTACGTAGCTAGCTAGCTAGCTAGC", height=150)

if st.button("Identify Genes"):
    # Load the trained 'brain'
    model = GeneDetector()
    try:
        model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
        model.eval()
        
        # AI Analysis
        input_tensor = torch.tensor(one_hot_encode(user_dna)).float().T.unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence = probs[0][1].item() * 100
            prediction = torch.argmax(output, dim=1).item()

        # Display Results
        if prediction == 1:
            st.success(f"✅ Gene Detected! AI Confidence: {confidence:.2f}%")
            
            # Run NCBI Identification with a spinner
            with st.spinner("Searching NCBI Global Database..."):
                gene_details = identify_gene_function(user_dna)
            
            st.subheader("Biological Identity & Function")
            st.info(gene_details)
            
            # CSV Download
            df = pd.DataFrame([{"Sequence": user_dna, "Confidence": f"{confidence:.2f}%", "Details": gene_details}])
            st.download_button("📥 Download Result (CSV)", df.to_csv(index=False), "gene_analysis.csv")
        else:
            st.error(f"❌ No Gene Detected. AI Confidence: {confidence:.2f}%")
            
    except FileNotFoundError:
        st.error("Error: 'gene_detector_model.pth' not found. Please upload it to your GitHub repository.")
