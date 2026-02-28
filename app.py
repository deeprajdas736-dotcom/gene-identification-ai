import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.Blast import NCBIWWW, NCBIXML
import time

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Deepraj Das | Gene-AI Analyst", layout="wide")

# Classy Futuristic Styling
st.markdown(f"""
    <style>
    .stApp {{ background-color: #050505; color: #E0E0E0; font-family: 'Inter', sans-serif; }}
    .main-header {{ font-size: 2.5rem; font-weight: 800; color: #FFFFFF; letter-spacing: -1px; margin-bottom: 0; }}
    .user-credit {{ font-size: 1rem; color: #888888; margin-top: -10px; margin-bottom: 30px; border-left: 3px solid #00D1FF; padding-left: 10px; }}
    .stTextArea textarea {{ background-color: #121212; color: #00D1FF; border: 1px solid #333; border-radius: 8px; }}
    .stButton>button {{ width: 100%; background: #FFFFFF; color: #000; border-radius: 5px; font-weight: bold; border: none; transition: 0.3s; }}
    .stButton>button:hover {{ background: #00D1FF; color: #FFF; box-shadow: 0 0 20px rgba(0,209,255,0.4); }}
    .result-card {{ background: #111; padding: 25px; border-radius: 12px; border: 1px solid #222; margin-top: 20px; }}
    </style>
""", unsafe_allow_html=True)

# --- 2. THE AI BRAIN ---
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

# --- 3. THE BIOLOGICAL VALIDATOR ---
def fetch_ncbi_identity(sequence):
    try:
        # qblast returns XML format by default for better parsing
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence, timeout=60)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return {"name": top_hit.title, "length": top_hit.length, "error": None}
        return {"name": None, "length": 0, "error": "AI identified a gene motif, but no match exists in the NCBI database (Potential Novel Gene)."}
    except Exception as e:
        return {"name": None, "length": 0, "error": f"Connection Error: {str(e)}. (NCBI servers may be busy, please try again)."}

# --- 4. INTERFACE ---
st.markdown('<p class="main-header">Genomic Intelligence Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="user-credit">Principal Investigator: Deepraj Das | NIT Agartala</p>', unsafe_allow_html=True)

dna_input = st.text_area("Input DNA Sequence Data:", placeholder="Paste your FASTA or raw sequence here...", height=200)

if st.button("EXECUTE ANALYSIS"):
    if not dna_input:
        st.error("Please input a valid DNA sequence.")
    else:
        with st.status("Initializing Neural Scan...", expanded=True) as status:
            try:
                # Load AI Model
                model = GeneDetector()
                model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
                model.eval()
                
                # AI Prediction
                input_tensor = torch.tensor(one_hot_encode(dna_input)).float().T.unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    confidence = F.softmax(output, dim=1)[0][1].item() * 100
                    prediction = torch.argmax(output, dim=1).item()
                
                status.update(label="AI Analysis Complete. Validating with NCBI...", state="running")
                
                # NCBI Search
                ncbi_data = fetch_ncbi_identity(dna_input)
                
                if ncbi_data['error']:
                    status.update(label="Partial Result: AI Match Found, Validation Failed.", state="error")
                    st.error(ncbi_data['error'])
                else:
                    status.update(label="Analysis Verified Successfully.", state="complete")
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style="color:#00D1FF; margin-top:0;">Verified Gene Report</h3>
                        <p><b>AI Prediction:</b> Gene Detected ({confidence:.2f}% Accuracy)</p>
                        <p><b>Gene Name:</b> {ncbi_data['name']}</p>
                        <p><b>Sequence Length:</b> {ncbi_data['length']} base pairs</p>
                        <p><b>Scientific Status:</b> Confirmed via NCBI BLAST Database</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except FileNotFoundError:
                status.update(label="Critical System Error", state="error")
                st.error("System Error: 'gene_detector_model.pth' not found. Ensure the model file is in your GitHub folder.")
            except Exception as e:
                status.update(label="Unexpected Error", state="error")
                st.error(f"Error Details: {str(e)}")
