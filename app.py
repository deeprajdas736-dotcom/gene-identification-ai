import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.Blast import NCBIWWW, NCBIXML

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Deepraj Das | Gene-AI Analyst", layout="wide")

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

# --- 2. AI MODEL (1D CNN) ---
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

# --- 3. BIOLOGICAL VALIDATION (NCBI BLAST) ---
def fetch_verified_data(sequence):
    try:
        # Step 1: Connect to NCBI (Timeout keyword removed to prevent error)
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        
        # Step 2: Parse results for the top hit
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            # Extracting name and length for 100% accuracy
            return {"name": top_hit.title, "length": top_hit.length, "error": None}
        return {"name": None, "length": 0, "error": "AI motif detected, but no matching sequence found in NCBI database (Potential Novel Gene/IDP)."}
    
    except Exception as e:
        # Standard error handling for connection issues
        return {"name": None, "length": 0, "error": "NCBI Server is currently busy or connection timed out. Please try again in 30 seconds."}

# --- 4. INTERFACE & LOGIC ---
st.markdown('<p class="main-header">Genomic Intelligence Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="user-credit">Principal Investigator: Deepraj Das | NIT Agartala</p>', unsafe_allow_html=True)

dna_input = st.text_area("Input DNA Sequence Data:", placeholder="Paste DNA sequence here...", height=200)

if st.button("EXECUTE NEURAL ANALYSIS"):
    if not dna_input:
        st.error("Error: Please provide a sequence for analysis.")
    else:
        with st.status("Analyzing Genomic Structure...", expanded=True) as status:
            try:
                # Load the 'Brain'
                model = GeneDetector()
                model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
                model.eval()
                
                # Neural Prediction
                input_tensor = torch.tensor(one_hot_encode(dna_input)).float().T.unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    confidence = F.softmax(output, dim=1)[0][1].item() * 100
                    prediction = torch.argmax(output, dim=1).item()
                
                if prediction == 1:
                    status.update(label="Gene Motif Found. Validating Identity...", state="running")
                    
                    # Real-world Validation
                    validation = fetch_verified_data(dna_input)
                    
                    if validation['error']:
                        status.update(label="Partial Success: AI match found.", state="error")
                        st.warning(validation['error'])
                    else:
                        status.update(label="Sequence Verified.", state="complete")
                        st.markdown(f"""
                        <div class="result-card">
                            <h3 style="color:#00D1FF; margin-top:0;">Verified Biological Report</h3>
                            <p><b>AI Status:</b> Gene Identified ({confidence:.2f}% Confidence)</p>
                            <p><b>Official Gene Name:</b> {validation['name']}</p>
                            <p><b>Sequence Length:</b> {validation['length']} bp</p>
                            <p><b>Validation Method:</b> Cloud-Linked NCBI BLASTn</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    status.update(label="Analysis Complete: Non-coding Region.", state="complete")
                    st.info(f"The AI did not detect significant gene motifs in this sequence (Confidence: {100-confidence:.2f}%).")
                    
            except Exception as e:
                status.update(label="System Error", state="error")
                st.error(f"Operational Error: {str(e)}")
