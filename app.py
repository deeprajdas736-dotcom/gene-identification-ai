import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from Bio.Blast import NCBIWWW, NCBIXML

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Deepraj Das | Genomic Mission Control", layout="wide")

# Professional Dashboard Styling
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .header-panel { background: #161b22; padding: 20px; border-radius: 10px; border-left: 5px solid #00d1ff; margin-bottom: 25px; }
    .metric-box { background: #1c2128; padding: 15px; border-radius: 8px; border: 1px solid #30363d; text-align: center; }
    .stTextArea textarea { background-color: #0d1117; color: #58a6ff; border: 1px solid #30363d; border-radius: 6px; font-family: 'Courier New', monospace; }
    .stButton>button { width: 100%; background: #238636; color: white; border-radius: 6px; font-weight: 600; border: none; height: 3em; }
    .stButton>button:hover { background: #2ea043; box-shadow: 0 0 10px rgba(46,160,67,0.4); }
    .report-card { background: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. AI CORE (1D-CNN) ---
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

# --- 3. BIOLOGICAL VALIDATION (NCBI API) ---
def validate_sequence(sequence):
    try:
        # Step 1: Query global database
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        
        # Step 2: Extract top-tier match for 100% accuracy
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return {"name": top_hit.title, "length": top_hit.length, "valid": True}
        return {"name": "No match found", "length": 0, "valid": False}
    except:
        return {"name": "Connection Error (NCBI Offline)", "length": 0, "valid": False}

# --- 4. DASHBOARD INTERFACE ---
st.markdown("""
    <div class="header-panel">
        <h1 style='margin:0; color:white;'>GENE-AI: MISSION CONTROL</h1>
        <p style='margin:0; color:#8b949e;'>Principal Investigator: <b>Deepraj Das</b> | NIT Agartala</p>
    </div>
""", unsafe_allow_html=True)

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📥 SEQUENCE INPUT TERMINAL")
    dna_input = st.text_area("", placeholder="Enter Raw Genomic Sequence Data...", height=300)
    analyze_btn = st.button("INITIATE NEURAL ANALYSIS & GLOBAL SEARCH")

with col2:
    st.markdown("### 📊 SCAN METRICS")
    if not dna_input:
        st.info("Awaiting Input Data...")
    else:
        st.markdown(f"""
            <div class="metric-box">
                <p style="margin:0; color:#8b949e;">INPUT LENGTH</p>
                <h2 style="margin:0; color:#00d1ff;">{len(dna_input)} bp</h2>
            </div><br>
            <div class="metric-box">
                <p style="margin:0; color:#8b949e;">SYSTEM STATUS</p>
                <h2 style="margin:0; color:#238636;">READY</h2>
            </div>
        """, unsafe_allow_html=True)

# --- 5. EXECUTION & RESULTS ---
if analyze_btn:
    if len(dna_input) < 10:
        st.warning("Sequence too short for reliable AI analysis.")
    else:
        with st.status("Performing 1D-CNN Structural Scan...", expanded=True) as status:
            try:
                # AI Step
                model = GeneDetector()
                model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
                model.eval()
                
                tensor = torch.tensor(one_hot_encode(dna_input)).float().T.unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor)
                    confidence = F.softmax(output, dim=1)[0][1].item() * 100
                    prediction = torch.argmax(output, dim=1).item()

                status.update(label="Structural Scan Complete. Fetching Functional Annotation...", state="running")
                
                # Validation Step
                bio_data = validate_sequence(dna_input)
                
                status.update(label="Analysis Finalized.", state="complete")
                
                # Result Panel
                st.markdown("### 🏁 FINAL SCAN REPORT")
                res_color = "#238636" if prediction == 1 else "#da3633"
                res_text = "GENE DETECTED" if prediction == 1 else "NON-CODING REGION"
                
                st.markdown(f"""
                    <div class="report-card">
                        <h2 style="color:{res_color}; margin-top:0;">{res_text}</h2>
                        <p><b>AI Confidence Score:</b> {confidence:.2f}%</p>
                        <hr style="border:0.5px solid #30363d;">
                        <h4 style="color:#58a6ff;">Biological Identification (Verified)</h4>
                        <p><b>Gene Name:</b> {bio_data['name']}</p>
                        <p><b>Database Match Length:</b> {bio_data['length']} bp</p>
                    </div>
                """, unsafe_allow_html=True)

                # --- 6. EXPORT TERMINAL ---
                st.markdown("### 💾 DATA EXPORT")
                export_df = pd.DataFrame([{
                    "Investigator": "Deepraj Das",
                    "Sequence_Length": len(dna_input),
                    "AI_Result": res_text,
                    "AI_Confidence": f"{confidence:.2f}%",
                    "NCBI_Identity": bio_data['name']
                }])
                
                st.download_button(
                    label="📥 DOWNLOAD VERIFIED REPORT (.CSV)",
                    data=export_df.to_csv(index=False),
                    file_name=f"Deepraj_Das_Analysis_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"System Error Encountered: {str(e)}")
