import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.Blast import NCBIWWW, NCBIXML

# 1. Define the AI Structure
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

# 2. Identification Function (NCBI Search)
def identify_gene_function(sequence):
    with st.spinner("Searching NCBI databases for gene name and function..."):
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return top_hit.title
        return "Unknown sequence (No match in NCBI database)."

# 3. Web Interface
st.title("🧬 Gene Identification & Functional Analysis")
user_dna = st.text_area("Paste DNA Sequence:", "ATGCGTACGTAGCTAGCTAGCTAGCTAGC")

if st.button("Identify Genes"):
    # Load AI
    model = GeneDetector()
    model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # AI Prediction
    input_tensor = torch.tensor(one_hot_encode(user_dna)).float().T.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence = probs[0][1].item() * 100
        prediction = torch.argmax(output, dim=1).item()

    # Display Results
    if prediction == 1:
        st.success(f"Gene Detected! AI Confidence: {confidence:.2f}%")
        # Run Identification
        gene_details = identify_gene_function(user_dna)
        st.subheader("Biological Identity & Function")
        st.info(gene_details)
    else:
        st.error(f"No Gene Detected. AI Confidence: {confidence:.2f}%")
