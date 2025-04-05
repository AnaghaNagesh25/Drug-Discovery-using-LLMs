import gradio as gr
import random
import base64
from io import BytesIO
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torch
import py3Dmol

# Load NLP & ML models
bio_gpt = pipeline("text-generation", model="microsoft/BioGPT-Large")
chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chemberta_model = AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
compliance_qa = pipeline("question-answering", model="nlpaueb/legal-bert-base-uncased")

def extract_insights(disease, symptoms):
    prompt = f"Recent treatments for {disease} with symptoms: {symptoms}"
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
        return buf.getvalue()
    except Exception as e:
        return f"Error visualizing molecule: {str(e)}"

def generate_3d_structure(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({"stick": {}})
        viewer.zoomTo()
        return viewer._make_html()
    except Exception as e:
        return f"Error generating 3D molecule: {str(e)}"

def check_compliance():
    try:
        question = "What does FDA require for drug testing?"
        context = "FDA requires extensive testing for new drug candidates including Phase I, II, and III clinical trials."
        return compliance_qa(question=question, context=context)['answer']
    except Exception as e:
        return f"Error checking compliance: {str(e)}"

def full_pipeline(disease, symptoms):
    insights = extract_insights(disease, symptoms)
    smiles = generate_molecule()
    mol_img = visualize_molecule(smiles)
    score = predict_properties(smiles)
    mol_3d_html = generate_3d_structure(smiles)
    compliance = check_compliance()
    return insights, mol_img, f"{smiles} | Score: {score}", mol_3d_html, compliance

demo = gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Textbox(label="Disease", value="lung cancer"),
        gr.Textbox(label="Symptoms", value="shortness of breath, weight loss")
    ],
    outputs=[
        gr.Textbox(label="Literature Insights"),
        gr.Image(label="2D Molecule"),
        gr.Textbox(label="Molecule Info"),
        gr.HTML(label="3D Molecule Structure"),
        gr.Textbox(label="Compliance Info")
    ],
    title="🧬 AI-Driven Drug Discovery",
    description="Enter a disease and symptoms to generate and analyze potential drug candidates using AI."
)

if __name__ == "__main__":
    demo.launch()
