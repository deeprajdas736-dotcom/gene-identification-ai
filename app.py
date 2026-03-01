import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from Bio.Blast import NCBIWWW, NCBIXML

# --- 1. RESEARCH STATION CONFIGURATION ---
st.set_page_config(page_title="Deepraj Das | GeneLab AI Pro", layout="wide")

# High-Fidelity UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(90deg, #1f6feb, #111); padding: 25px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px; }
    .stat-card { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    .stButton>button { width: 100%; background-color: #238636; color: white; border-radius: 8px; font-weight: 700; border: none; padding: 10px; }
    .report-container { background: #161b22; padding: 25px; border-radius: 12px; border: 1px solid #1f6feb; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. NEURAL CORE ---
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

def fetch_ncbi_data(sequence):
    try:
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return {"name": top_hit.title, "length": top_hit.length, "valid": True}
        return {"name": "No Database Match Found (Potential Novel Gene)", "length": 0, "valid": False}
    except:
        return {"name": "NCBI Server Busy - Analysis Incomplete", "length": 0, "valid": False}

# --- 3. DASHBOARD INTERFACE ---
with st.sidebar:
    st.markdown("### 🧬 RESEARCHER PROFILE")
    st.write("**Deepraj Das**")
    st.write("MSc Biotechnology")
    st.write("NIT Agartala")
    st.markdown("---")
    st.write("Target: IDP/Genomic Identification")

st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>GENELAB PRO: MISSION CONTROL</h1>
        <p style='margin:0; opacity:0.8;'>Automated Genomic Identification Platform</p>
    </div>
""", unsafe_allow_html=True)

# Metric Panel
col1, col2, col3 = st.columns(3)
col1.markdown("<div class='stat-card'><p style='color:#8b949e;'>CORE MODEL</p><h3 style='color:#1f6feb;'>1D-CNN</h3></div>", unsafe_allow_html=True)
col2.markdown("<div class='stat-card'><p style='color:#8b949e;'>DB LINK</p><h3 style='color:#3fb950;'>NCBI LIVE</h3></div>", unsafe_allow_html=True)
col3.markdown("<div class='stat-card'><p style='color:#8b949e;'>SYSTEM STATUS</p><h3 style='color:#00d1ff;'>ACTIVE</h3></div>", unsafe_allow_html=True)

st.write("---")
dna_input = st.text_area("📡 SEQUENCE INPUT TERMINAL", placeholder="Paste Nucleotide Sequence...", height=250)
run_analysis = st.button("🚀 EXECUTE MULTI-LAYER SCAN")

if run_analysis:
    if not dna_input:
        st.error("Input Stream Empty.")
    else:
        with st.status("Performing Structural Scan...", expanded=True) as status:
            try:
                # 1. Neural Prediction
                model = GeneDetector()
                model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
                model.eval()
                
                tensor = torch.tensor(one_hot_encode(dna_input)).float().T.unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor)
                    confidence = F.softmax(output, dim=1)[0][1].item() * 100
                    prediction = torch.argmax(output, dim=1).item()
                
                status.update(label="Structural Scan Complete. Validating via NCBI...", state="running")
                
                # 2. Biological Validation
                bio_data = fetch_ncbi_data(dna_input)
                status.update(label="Analysis Finalized.", state="complete")
                
                # 3. Informative Tabs
                res_tab, blast_tab, export_tab = st.tabs(["📊 Neural Report", "🧬 Verified Identity", "📥 Data Export"])
                
                with res_tab:
                    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                    res_txt = "GENE DETECTED" if prediction == 1 else "NON-CODING"
                    res_clr = "#3fb950" if prediction == 1 else "#f85149"
                    st.markdown(f"<h2 style='color:{res_clr};'>{res_txt}</h2>", unsafe_allow_html=True)
                    st.write(f"**AI Confidence Level:** {confidence:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with blast_tab:
                    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                    st.subheader("Biological Annotation")
                    st.info(f"**Identification:** {bio_data['name']}")
                    st.write(f"**Confirmed Length:** {bio_data['length']} bp")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with export_tab:
                    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                    st.subheader("Export Results")
                    df = pd.DataFrame([{"Investigator": "Deepraj Das", "Result": res_txt, "Confidence": f"{confidence:.2f}%", "Identity": bio_data['name']}])
                    st.download_button(
                        label="📥 DOWNLOAD VERIFIED REPORT (.CSV)",
                        data=df.to_csv(index=False),
                        file_name=f"GeneLab_Report_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Critical operational failure: {str(e)}")
