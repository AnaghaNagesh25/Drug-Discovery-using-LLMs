import sys
import subprocess

# Force package path in case it's misconfigured
sys.path.append("/home/adminuser/venv/lib/python3.12/site-packages")

# Verify package install
try:
    import transformers
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "--no-cache-dir", "transformers", "torch"])
    import transformers  # Retry after install

# Now import everything else
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import py3Dmol
import torch
import random
import base64
from io import BytesIO

# Load NLP Models
bio_gpt = pipeline("text-generation", model="microsoft/BioGPT-Large")
chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chemberta_model = AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
compliance_qa = pipeline("question-answering", model="nlpaueb/legal-bert-base-uncased")

# Streamlit UI
st.title("🧬 AI-Driven Drug Discovery System")
st.write("Enter disease & symptoms to generate potential drug candidates.")

# User Input
disease = st.text_input("Disease:", "lung cancer")
symptoms = st.text_area("Symptoms:", "shortness of breath, weight loss")

# Function: Extract Medical Insights
def extract_insights(prompt):
    try:
        result = bio_gpt(prompt, max_length=200, do_sample=True)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error extracting insights: {str(e)}"

# Function: Generate Random Molecule (SMILES)
def generate_molecule():
    sample_smiles = ["CCO", "CCN", "C1=CC=CC=C1", "C(C(=O)O)N", "CC(C)CC"]
    return random.choice(sample_smiles)

# Function: Predict Molecular Properties using ChemBERTa
def predict_properties(smiles):
    try:
        inputs = chemberta_tokenizer(smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = chemberta_model(**inputs)
        return outputs.logits.mean().item()
    except Exception as e:
        return f"Error predicting properties: {str(e)}"

# Function: Visualize Molecule in 2D
def visualize_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        return f"Error generating image: {str(e)}"

# Function: Generate 3D Molecule Structure
def generate_3d_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block
    except Exception as e:
        return f"Error generating 3D molecule: {str(e)}"

# Function: Compliance Check using LegalBERT
def check_compliance(question, context):
    try:
        return compliance_qa(question=question, context=context)['answer']
    except Exception as e:
        return f"Error checking compliance: {str(e)}"

# Processing the Inputs
if st.button("Discover Drug Candidates"):
    with st.spinner("🔍 Analyzing biomedical literature..."):
        insights = extract_insights(f"Recent treatment options for {disease} with symptoms {symptoms}")

    with st.spinner("🧪 Generating new molecule..."):
        smiles = generate_molecule()
        image_data = visualize_molecule(smiles)
        property_score = predict_properties(smiles)
        mol_3d = generate_3d_molecule(smiles)

    with st.spinner("✅ Checking legal compliance..."):
        compliance_info = check_compliance("What does FDA require for drug testing?",
                                           "FDA requires extensive testing for new drug candidates including Phase I, II, and III clinical trials.")

    # Display Insights
    st.subheader("📜 Literature Insights")
    st.write(insights)

    # Display Molecule
    st.subheader("🧪 Generated Molecule")
    st.image(f"data:image/png;base64,{image_data}", caption=f"SMILES: {smiles}")

    # Display Molecular Properties
    st.subheader("🧬 Molecular Properties")
    st.write(f"Predicted Property Score: {property_score}")

    # Display 3D Molecule
    st.subheader("🔬 3D Molecule Structure")
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mol_3d, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    st.components.v1.html(viewer.show(), height=400)

    # Display Compliance Info
    st.subheader("⚖️ Legal Compliance")
    st.write(compliance_info)
