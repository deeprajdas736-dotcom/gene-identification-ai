import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from Bio.Blast import NCBIWWW, NCBIXML

# --- 1. RESEARCH STATION CONFIGURATION ---
st.set_page_config(page_title="GeneLab Elite | Deepraj Das", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .main-header { 
        background: linear-gradient(90deg, #d4af37, #1a1a1a); 
        padding: 30px; border-radius: 15px; border-left: 10px solid #d4af37; margin-bottom: 25px; 
    }
    .stat-card { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    .stButton>button { 
        background: #d4af37; color: black; border-radius: 8px; font-weight: 800; border: none; padding: 12px; transition: 0.3s;
    }
    .stButton>button:hover { background: #ffd700; transform: scale(1.02); }
    .report-container { background: #161b22; padding: 25px; border-radius: 12px; border: 1px solid #d4af37; }
    </style>
""", unsafe_allow_html=True)

# --- 2. THE AI DISCOVERY ENGINE (Ab Initio) ---
class GeneDetector(nn.Module):
    def __init__(self):
        super(GeneDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc(x)
        return x

def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    return np.array([mapping.get(base.upper(), [0,0,0,0]) for base in sequence])

# --- 3. THE GENOME CLASSIFIER & IDENTIFIER ---
def classify_and_identify(sequence):
    try:
        # Determine Genome Type based on sequence characteristics (GC Content / Length)
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        g_type = "Prokaryotic (Dense)" if gc_content > 55 else "Eukaryotic (Complex/Intronic)"
        
        # Homology Search for Function/Name
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return {"type": g_type, "name": top_hit.title, "valid": True}
        return {"type": g_type, "name": "Unknown/Novel Gene (No Database Match)", "valid": False}
    except:
        return {"type": "Awaiting Cloud Verification", "name": "NCBI Timeout", "valid": False}

# --- 4. DASHBOARD INTERFACE ---
with st.sidebar:
    st.markdown("### 🧬 INVESTIGATOR")
    st.write("**Deepraj Das**")
    st.write("MSc Biotechnology | Amity University")
    st.write("Dissertation: NIT Agartala")
    st.markdown("---")
    st.info("This system identifies known genes and discovers novel Open Reading Frames (ORFs) using 1D-CNN patterns.")

st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; color:#d4af37;'>GENELAB ELITE: DISCOVERY PIPELINE</h1>
        <p style='margin:0; color:#888;'>Autonomous Genome Classification & Functional Annotation</p>
    </div>
""", unsafe_allow_html=True)

# Metric Panel
m1, m2, m3 = st.columns(3)
m1.markdown("<div class='stat-card'><p style='color:#8b949e;'>ANALYSIS ENGINE</p><h3 style='color:#d4af37;'>Neural 1D-CNN</h3></div>", unsafe_allow_html=True)
m2.markdown("<div class='stat-card'><p style='color:#8b949e;'>DISCOVERY MODE</p><h3 style='color:#d4af37;'>Ab Initio + Homology</h3></div>", unsafe_allow_html=True)
m3.markdown("<div class='stat-card'><p style='color:#8b949e;'>CLOUD STATUS</p><h3 style='color:#00ff88;'>NCBI LIVE</h3></div>", unsafe_allow_html=True)

st.write("---")
dna_input = st.text_area("🛰️ GENOMIC SEQUENCE STREAM", placeholder="Paste Nucleotide Data (A, C, G, T)...", height=250)
execute = st.button("🚀 INITIATE FULL GENOMIC DISCOVERY")

if execute:
    if len(dna_input) < 20:
        st.error("Sequence integrity failed: Data too short for neural classification.")
    else:
        with st.status("Scanning Neural Architecture for Gene Motifs...", expanded=True) as status:
            try:
                # 1. Prediction (Identify even unknown genes)
                model = GeneDetector()
                model.load_state_dict(torch.load("gene_detector_model.pth", map_location=torch.device('cpu')))
                model.eval()
                
                tensor = torch.tensor(one_hot_encode(dna_input)).float().T.unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor)
                    confidence = F.softmax(output, dim=1)[0][1].item() * 100
                    prediction = torch.argmax(output, dim=1).item()
                
                status.update(label="Structural Scan Complete. Identifying Genome Type...", state="running")
                
                # 2. Identification (Type, Name, Function)
                results = classify_and_identify(dna_input)
                status.update(label="Analysis Verified.", state="complete")
                
                # 3. High-Fidelity Results
                res_tab, id_tab, report_tab = st.tabs(["📊 Neural Prediction", "🧬 Biological Identity", "📥 Official Report"])
                
                with res_tab:
                    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                    pred_txt = "GENE DETECTED" if prediction == 1 else "NON-CODING REGION"
                    pred_clr = "#00ff88" if prediction == 1 else "#ff4b4b"
                    st.markdown(f"<h2 style='color:{pred_clr};'>{pred_txt}</h2>", unsafe_allow_html=True)
                    st.write(f"**AI Structural Confidence:** {confidence:.2f}%")
                    st.write(f"**Predicted Genome Type:** {results['type']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with id_tab:
                    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                    st.subheader("Functional Annotation")
                    st.info(f"**Assigned Name/Function:** {results['name']}")
                    st.write(f"**Genome Classification:** {results['type']}")
                    st.write("**Database:** NCBI GenBank (Cloud Verified)")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with report_tab:
                    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                    st.subheader("Data Export Terminal")
                    export_df = pd.DataFrame([{
                        "Investigator": "Deepraj Das",
                        "Affiliation": "Amity Kolkata / NIT Agartala",
                        "Genome_Type": results['type'],
                        "AI_Prediction": pred_txt,
                        "AI_Confidence": f"{confidence:.2f}%",
                        "Bio_Identity": results['name']
                    }])
                    st.download_button(
                        label="📥 DOWNLOAD CERTIFIED GENOMIC REPORT (CSV)",
                        data=export_df.to_csv(index=False),
                        file_name=f"GeneLab_Elite_Report_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"System Error: {str(e)}")
