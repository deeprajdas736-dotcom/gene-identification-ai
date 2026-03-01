import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from Bio.Blast import NCBIWWW, NCBIXML

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Deepraj Das | GeneLab AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(90deg, #1f6feb, #111); padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px; }
    .stat-card { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    .report-container { background: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #1f6feb; margin-top: 10px; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; font-weight: 700; border: none; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. AI CORE ---
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
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            top_hit = blast_record.alignments[0]
            return {"name": top_hit.title, "length": top_hit.length, "match": True}
        return {"name": "No Database Match Found", "length": 0, "match": False}
    except:
        return {"name": "NCBI Connection Timeout", "length": 0, "match": False}

# --- 3. DASHBOARD ---
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>GENELAB AI: RESEARCH DASHBOARD</h1>
        <p style='margin:0; opacity:0.8;'>Principal Investigator: <b>Deepraj Das</b> | NIT Agartala</p>
    </div>
""", unsafe_allow_html=True)

# Metric Row
m1, m2, m3 = st.columns(3)
m1.markdown("<div class='stat-card'><p style='color:#8b949e;'>MODEL</p><h3 style='color:#1f6feb;'>1D-CNN</h3></div>", unsafe_allow_html=True)
m2.markdown("<div class='stat-card'><p style='color:#8b949e;'>DB LINK</p><h3 style='color:#3fb950;'>NCBI LIVE</h3></div>", unsafe_allow_html
