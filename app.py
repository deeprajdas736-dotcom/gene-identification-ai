import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from Bio.Blast import NCBIWWW, NCBIXML

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Deepraj Das | GeneLab AI", layout="wide", initial_sidebar_state="expanded")

# Advanced UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .main-header { background: linear-gradient(90deg, #1f6feb, #111); padding: 25px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px; }
    .stat-card { background: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    .report-container { background: #0d1117; padding: 25px; border-radius: 12px; border: 1px solid #1f6feb; margin-top: 20px; box-shadow: 0 4px 20px rgba(31,111,235,0.1); }
    .stButton>button { width: 100%; background-color: #238636; color: white; border-radius: 8px; font-weight: 700; border: none; padding: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .stButton>button:hover { background-color: #2ea043; border: 1px solid #3fb950; box-shadow: 0 0 15px rgba(46,160,67,0.4); }
    h1, h2, h3 { color: #f0f6fc; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE BIOLOGICAL ENGINE ---
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

def run_blast(sequence):
    try:
        # standard qblast call without timeout keyword to avoid TypeError
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return {"name": top_hit.title, "length": top_hit.length, "match": True}
        return {"name": "Novel Sequence Detected (No Database Match)", "length": 0, "match": False}
    except:
        return {"name": "NCBI Server Timeout - Local Prediction Mode Only", "length": 0, "match": False}

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png")
    st.title("🧬 GeneLab AI")
    st.markdown("---")
    st.subheader("RESEARCHER PROFILE")
    st.write("**Deepraj Das**")
    st.write("MSc Biotechnology")
    st.write("NIT Agartala")
    st.markdown("---")
    st.subheader("SCAN PARAMETERS")
    sensitivity = st.slider("Neural Sensitivity (%)", 50, 99, 80)
    st.info("Currently optimized for Intrinsically Disordered Proteins (IDPs).")

# --- 4. MAIN DASHBOARD ---
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>GENELAB: INTEGRATED GENOMIC PLATFORM</h1>
        <p style='margin:0; color:#8b949e; opacity:0.8;'>Automated Gene Identification & Functional Annotation Engine</p>
    </div>
""", unsafe_allow_html=True)

# Hero Section: Metrics & Input
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown("<div class='stat-card'><p style='color:#8b949e;'>SYSTEM STATUS</p><h2 style='color:#3fb950;'>ONLINE</h2></div>", unsafe_allow_html=True)
with m_col2:
    st.markdown("<div class='stat-card'><p style='color:#8b949e;'>ACTIVE MODEL</p><h2 style='color:#1f6feb;'>1D-CNN v2.1</h2></div>", unsafe_allow_html=True)
with m_col3:
    st.markdown("<div class='stat-card'><p style='color:#8b949e;'>DATABASE LINK</p><h2 style='color:#f85149;'>NCBI CLOUD</h2></div>", unsafe_allow_html=True)

st.write("---")

input_area, settings_area = st.columns([2.5, 1])

with input_area:
    st.subheader("🧬 Sequence Input Terminal")
    dna_input = st.text_area("", placeholder="Paste Nucleotide Sequence (A, C, G, T) for Deep Neural Analysis...", height=250)
    execute = st.button("🚀 EXECUTE FULL GENOMIC ANALYSIS")

with settings_area:
    st.subheader("📡 Analysis Logs")
    if not dna_input:
        st.write("Waiting for data stream...")
    else:
        st.write(f"• Input Stream: {len(dna_input)} bp received")
        st.write(f"• Encoding: One-Hot Matrix")
        st.write(f"• Threads: Parallel Neural Processing")

# --- 5. EXECUTION & VISUALIZATION ---
if execute:
    if len(dna_input) < 20:
        st.error("Sequence integrity check failed. Minimum 20bp required.")
    else:
        with st.status("Initializing Neural Architecture...", expanded=True) as status:
            try:
                # AI Prediction Phase
                model = GeneDetector()
                model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
                model.eval()
                
                encoded = one_hot_encode(dna_input)
                tensor = torch.tensor(encoded).float().T.unsqueeze(0)
                
                with torch.no_grad():
                    output = model(tensor)
                    confidence = F.softmax(output, dim=1)[0][1].item() * 100
                    prediction = torch.argmax(output, dim=1).item()

                status.update(label="Structural Motifs Detected. Accessing NCBI Cloud...", state="running")
                
                # NCBI Validation Phase
                bio_data = run_blast(dna_input)
                
                status.update(label="Analysis Finalized.", state="complete")
                
                # tabs for organized results (mimicking your reference image)
                res_tab, blast_tab, export_tab = st.tabs(["📊 Neural Prediction", "
