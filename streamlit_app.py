import streamlit as st
import random
import base64
from io import BytesIO
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torch
import py3Dmol

# Load NLP & ML models
@st.cache_resource
def load_models():
    bio_gpt = pipeline("text-generation", model="microsoft/BioGPT-Large")
    chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    chemberta_model = AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    compliance_qa = pipeline("question-answering", model="nlpaueb/legal-bert-base-uncased")
    return bio_gpt, chemberta_tokenizer, chemberta_model, compliance_qa

bio_gpt, chemberta_tokenizer, chemberta_model, compliance_qa = load_models()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Drug Discovery", layout="centered")
st.title("🧬 AI-Driven Drug Discovery System")
st.markdown("Enter disease & symptoms to generate potential drug candidates.")

disease = st.text_input("🦠 Disease:", "lung cancer")
symptoms = st.text_area("🩺 Symptoms:", "shortness of breath, weight loss")

# --- Functions ---
def extract_insights(prompt):
    try:
        result = bio_gpt(prompt, max_length=200, do_sample=True)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error extracting insights: {str(e)}"

def generate_molecule():
    sample_smiles = ["CCO", "CCN", "C1=CC=CC=C1", "C(C(=O)O)N", "CC(C)CC"]
    return random.choice(sample_smiles)

def predict_properties(smiles):
    try:
        inputs = chemberta_tokenizer(smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = chemberta_model(**inputs)
        return round(outputs.logits.mean().item(), 3)
    except Exception as e:
        return f"Error predicting properties: {str(e)}"

def visualize_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return f"Error visualizing molecule: {str(e)}"

def generate_3d_structure(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        return f"Error generating 3D molecule: {str(e)}"

def check_compliance(question, context):
    try:
        return compliance_qa(question=question, context=context)['answer']
    except Exception as e:
        return f"Error checking compliance: {str(e)}"

# --- Run Pipeline ---
if st.button("🚀 Discover Drug Candidates"):
    with st.spinner("🔍 Analyzing biomedical literature..."):
        insights = extract_insights(f"Recent treatments for {disease} with symptoms: {symptoms}")

    with st.spinner("🧪 Generating molecule..."):
        smiles = generate_molecule()
        mol_image = visualize_molecule(smiles)
        mol_score = predict_properties(smiles)
        mol_3d = generate_3d_structure(smiles)

    with st.spinner("✅ Checking compliance..."):
        compliance = check_compliance("What does FDA require for drug testing?",
            "FDA requires extensive testing for new drug candidates including Phase I, II, and III clinical trials.")

    # Display Results
    st.subheader("📜 Literature Insights")
    st.write(insights)

    st.subheader("🧪 Generated Molecule")
    st.image(f"data:image/png;base64,{mol_image}", caption=f"SMILES: {smiles}")

    st.subheader("🧬 Molecular Properties")
    st.write(f"ChemBERTa Property Score: **{mol_score}**")

    st.subheader("🔬 3D Molecule Structure")
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mol_3d, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    st.components.v1.html(viewer.show(), height=400)

    st.subheader("⚖️ Legal Compliance")
    st.write(compliance)
