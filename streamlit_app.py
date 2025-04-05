import sys
import subprocess
import streamlit as st
import numpy as np
import pandas as pd
import torch
import random
import base64
from io import BytesIO

# Fix package path if needed
sys.path.append("/home/adminuser/venv/lib/python3.12/site-packages")

# Retry install if missing
try:
    import transformers
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "--no-cache-dir", "transformers", "torch"])
    import transformers

# NLP Models
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# RDKit Setup
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "--no-cache-dir", "rdkit-pypi"])
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

# 3Dmol Setup (For Streamlit, we embed it as HTML)
import py3Dmol

# Load NLP models
try:
    bio_gpt = pipeline("text-generation", model="microsoft/BioGPT-Large")
except:
    st.error("❌ Failed to load BioGPT model.")
    st.stop()

chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chemberta_model = AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
compliance_qa = pipeline("question-answering", model="nlpaueb/legal-bert-base-uncased")

# Streamlit UI
st.set_page_config(page_title="AI Drug Discovery", layout="wide")
st.title("🧬 AI-Driven Drug Discovery System")
st.write("Enter disease & symptoms to generate potential drug candidates.")

# Input
disease = st.text_input("Disease:", "lung cancer")
symptoms = st.text_area("Symptoms:", "shortness of breath, weight loss")

# Medical Insight Extraction
def extract_insights(prompt):
    try:
        result = bio_gpt(prompt, max_length=200, do_sample=True)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Generate random SMILES
def generate_molecule():
    sample_smiles = ["CCO", "CCN", "C1=CC=CC=C1", "C(C(=O)O)N", "CC(C)CC"]
    return random.choice(sample_smiles)

# Predict molecular properties
def predict_properties(smiles):
    try:
        inputs = chemberta_tokenizer(smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = chemberta_model(**inputs)
        return outputs.logits.mean().item()
    except Exception as e:
        return f"Error: {str(e)}"

# 2D visualization
def visualize_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        return f"Error: {str(e)}"

# 3D generation
def generate_3d_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block
    except Exception as e:
        return f"Error: {str(e)}"

# Legal compliance
def check_compliance(question, context):
    try:
        result = compliance_qa(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"

# Main execution
if st.button("🚀 Discover Drug Candidates"):
    with st.spinner("📖 Analyzing biomedical literature..."):
        insights = extract_insights(f"Recent treatment options for {disease} with symptoms {symptoms}")

    with st.spinner("🧪 Generating molecule and structure..."):
        smiles = generate_molecule()
        img_data = visualize_molecule(smiles)
        prop_score = predict_properties(smiles)
        mol_block = generate_3d_molecule(smiles)

    with st.spinner("⚖️ Checking compliance..."):
        compliance_info = check_compliance(
            "What does FDA require for drug testing?",
            "FDA requires extensive testing for new drug candidates including Phase I, II, and III clinical trials."
        )

    st.subheader("📜 Literature Insights")
    st.write(insights)

    st.subheader("🧪 Generated Molecule")
    st.image(f"data:image/png;base64,{img_data}", caption=f"SMILES: {smiles}")

    st.subheader("🧬 Molecular Property Score")
    st.write(f"Predicted Property Score: {prop_score:.3f}")

    st.subheader("🔬 3D Molecule Structure")
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    mol_html = viewer._make_html()
    st.components.v1.html(mol_html, height=400)

    st.subheader("⚖️ Legal Compliance Insight")
    st.write(compliance_info)
