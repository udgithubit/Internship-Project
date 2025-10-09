
# Toxic Content Detection in Hindi Social Media
Hybrid Transformer-Based Neural Architectures for Responsible NLP

## Overview
This project addresses the critical need for toxic content detection in Hindi social media platforms using advanced Natural Language Processing (NLP) methods. The work focuses on linguistic complexities unique to Hindi, such as code-mixing, dialectal diversity, and cultural context, and proposes hybrid neural architectures combining BERT (Bidirectional Encoder Representations from Transformers) and Bi-LSTM (Bidirectional Long Short-Term Memory) for robust classification.

## Key Features
- Data pipeline for collection, annotation, and preprocessing of Hindi social media posts
- Custom embeddings (Word2Vec, GloVe, FastText) and transformer-based features
- Hybrid deep learning model architecture (BERT + Bi-LSTM)
- Fairness evaluation using IBM AI Fairness 360 toolkit
- Published results in ICMLDE 2025: “Hybrid Transformer-Based Neural Architectures for Toxic Hindi Social Media”

## Repository Contents
- `data/`: Contains annotated Hindi social media dataset samples
- `notebooks/`: Python/Jupyter Notebooks for data processing, model building, evaluation
- `models/`: Pretrained embeddings and model weights
- `results/`: Performance metrics, classification reports, and visualization outputs
- `docs/`: Research paper, internship report, annexure, and project appendix

## Setup & Installation
1. Clone the repository:
   ```
   git clone < (https://github.com/udgithubit/Internship-Project/new/main)
   cd udgithubit
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Download required datasets (see `data/README.md` for instructions).
4. Start exploration in `notebooks/`.

## Major Outcomes
- Achieved 94% accuracy and 0.94 F1-score using hybrid BERT-BiLSTM architecture.
- Improved fairness (Equalized Odds, Demographic Parity, Individual Fairness).
- Model generalizes well to code-mixed and regional Hindi data.
- Comprehensive report and publication in ICMLDE 2025.

## How to Use
- Train the model or use provided weights for toxic content detection.
- Review fairness metrics to ensure responsible AI.
- Refer to `notebooks/Classification_Report.ipynb` for usage examples.
- For deployment, see instructions in `docs/deployment.md`.

## License
This project is open-sourced under the MIT License. See LICENSE file for details.

## Acknowledgments
- Dibrugarh University Institute of Engineering and Technology (DUIET)
- Amity University mentors and collaboration team
- Funding: DST WISE Fellowship
- Peer reviewers and contributors to ICMLDE 2025 publication

## Contact
For inquiries or collaboration, please reach out via dasunmona60@gmail.com. 

