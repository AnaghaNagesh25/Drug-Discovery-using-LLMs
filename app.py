import gradio as gr
from transformers import pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import base64
from io import BytesIO
import py3Dmol
import re

# Function to generate literature and 3D molecule view
def drug_discovery(disease, symptoms):
    # BioGPT pipeline
    bio_gpt = pipeline("text-generation", model="microsoft/BioGPT-Large")
    prompt = f"Recent treatments for {disease} with symptoms: {symptoms}."
    literature = bio_gpt(prompt, max_length=200)[0]['generated_text']

    # Generate SMILES using BioGPT with stricter filtering
    molecule_prompt = f"List 5 different valid drug-like SMILES strings that can treat {disease} with symptoms {symptoms}. Only list SMILES separated by spaces."
    smiles_result = bio_gpt(molecule_prompt, max_length=100)[0]['generated_text']

    # Extract and validate SMILES strings
    smiles_matches = re.findall(r"(?<![A-Za-z0-9])[A-Za-z0-9@+\-\[\]\(\)=#$]{5,}(?![A-Za-z0-9])", smiles_result)
    smiles = None
    for match in smiles_matches:
        mol_test = Chem.MolFromSmiles(match)
        if mol_test:
            smiles = match
            break
    if not smiles:
        smiles = "C1=CC=CC=C1"  # fallback to benzene if all fail

    # Generate RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES generated", smiles, "", ""

    AllChem.Compute2DCoords(mol)

    # Draw 2D image
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()

    # Convert binary to base64
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    img_html = f'''<div style="text-align:center; margin-top: 10px; animation: fadeIn 2s ease-in-out;">
        <img src="data:image/png;base64,{img_base64}" alt="2D Molecule" 
             style="border-radius: 16px; box-shadow: 0 6px 20px rgba(0,255,255,0.3); border: 1px solid #444;">
        <div style='font-family: Arial, sans-serif; color: #eeeeee; margin-top: 8px; animation: slideUp 1.5s ease-in-out;'>💊 Visualized Drug Molecule (2D)</div>
    </div>'''

    # 3D molecule
    mol3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol3d)
    AllChem.UFFOptimizeMolecule(mol3d)
    mb = Chem.MolToMolBlock(mol3d)

    viewer = py3Dmol.view(width=420, height=420)
    viewer.addModel(mb, "mol")
    viewer.setStyle({"stick": {"colorscheme": "cyanCarbon"}})
    viewer.setBackgroundColor("black")
    viewer.zoomTo()
    viewer.spin(True)
    viewer_html_raw = viewer._make_html()

    viewer_html = f'''
    <div style="text-align:center; margin-top: 20px; animation: zoomIn 2s ease-in-out;">
        <iframe srcdoc="{viewer_html_raw.replace('"', '&quot;')}" 
                width="440" height="440" frameborder="0" 
                style="border-radius: 16px; box-shadow: 0 8px 30px rgba(0,255,255,0.35);"></iframe>
        <div style='font-family: Arial, sans-serif; color: #eeeeee; margin-top: 8px; animation: slideUp 1.5s ease-in-out;'>🧬 Animated 3D Molecule (Stick View)</div>
    </div>'''

    return literature, smiles, img_html, viewer_html

# Gradio UI
disease_input = gr.Textbox(label="🏥 Enter Disease (e.g., lung cancer)", value="lung cancer")
symptom_input = gr.Textbox(label="💉 Enter Symptoms (e.g., cough, weight loss)", value="shortness of breath, weight loss")
lit_output = gr.Textbox(label="📰 Literature Insights from BioGPT")
smiles_output = gr.Textbox(label="🧪 SMILES Representation")
img_output = gr.HTML(label="🖼️ Molecule 2D Visualization")
viewer_output = gr.HTML(label="🔬 3D Drug Molecule Animation")

custom_css = """
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes slideUp {
    from {transform: translateY(40px); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}

@keyframes zoomIn {
    from {transform: scale(0.5); opacity: 0;}
    to {transform: scale(1); opacity: 1;}
}

body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: #eeeeee;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    animation: fadeIn 1.5s ease-in-out;
}

.gradio-container .block-label {
    color: #ffffff;
}
"""

iface = gr.Interface(
    fn=drug_discovery,
    inputs=[disease_input, symptom_input],
    outputs=[lit_output, smiles_output, img_output, viewer_output],
    title="🏥 AI-Powered Drug Discovery for Hospitals",
    description="This hospital-themed platform takes a disease and symptoms as input, retrieves biomedical insights using BioGPT, and visualizes potential drug molecules in 2D and animated 3D. Ideal for clinical research and pharma innovation.",
    theme="default",
    css=custom_css
)

iface.launch(share=True)
