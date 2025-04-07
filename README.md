# Drug-Discovery-using-LLMs
# 🧬 AI-Powered Drug Discovery using BioGPT + RDKit

> An AI-driven biomedical application that generates literature insights and visualizes drug-like molecules in 2D and 3D based on disease and symptom inputs. Built with BioGPT, RDKit, py3Dmol, and Gradio.

---

## 🚀 Live Demo

👉 [Try the App](https://anaghanagesh-drug-discovery-using-llms.hf.space/)

---

## 🧠 About the Project

This project leverages large language models and cheminformatics tools to:

- 🧾 Extract biomedical treatment insights using **BioGPT**
- 💊 Generate and validate **drug-like SMILES strings**
- 🧪 Visualize molecules in **2D (RDKit)** and **3D (py3Dmol)**
- 🖥️ Provide an interactive **Gradio-based user interface** for researchers and clinicians

---

## 💡 How It Works

1. **User Inputs**  
   - Disease name (e.g., *lung cancer*)
   - Symptoms (e.g., *shortness of breath, weight loss*)

2. **LLM Processing**  
   - BioGPT generates recent treatment literature  
   - Also suggests valid drug-like SMILES strings

3. **Molecular Visualization**  
   - SMILES is rendered in 2D using RDKit  
   - Molecule is animated in 3D using py3Dmol

---

## 🛠️ Technologies Used

- 🤖 `microsoft/BioGPT-Large` – Biomedical LLM
- 🧪 `RDKit` – Molecular representation and 2D drawing
- 🔬 `py3Dmol` – 3D stick model visualization
- 🌐 `Gradio` – Frontend & deployment
- 🐍 `Python 3.x`

---

## 📦 Installation

```bash
git clone https://github.com/anaghanagesh/drug-discovery-llm.git
cd drug-discovery-llm
pip install -r requirements.txt
python app.py
