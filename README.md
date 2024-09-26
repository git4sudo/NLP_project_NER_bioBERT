---

# NLP-Driven Analysis and Named Entity Recognition (NER) of Vaccine Adverse Events

This repository contains the implementation of a Natural Language Processing (NLP) project that utilizes BioBERT and Stanza to analyze 140,000 reports from the Vaccine Adverse Event Reporting System (VAERS). The goal is to extract and categorize medical terminologies related to adverse events following vaccinations, with a focus on improving healthcare investigations and vaccine safety analysis.

## Overview

This project is designed to automatically identify and extract reported symptoms from VAERS narratives and link them to standard medical terminologies. Leveraging a fine-tuned BioBERT model integrated with Stanza, we achieved the following key results:

- **Accuracy**: 97.9% in extracting medical terminologies.
- **F1 Score**: Increased by 18%, achieving a final score of 0.96.
- **Model Loss**: Reduced to 0.0776.

This work supports vaccine safety investigations by enhancing the analysis of clinical data in adverse event reports.

## Features

- **Named Entity Recognition (NER)**: Extracts symptom-related entities such as "fever" or "headache" from VAERS narratives.
- **Fine-tuning of BioBERT**: Optimized with clinical and biomedical embeddings to improve accuracy and efficiency.
- **NLP Techniques**: Utilizes sequence labeling for medical terminology extraction, improving accuracy in Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
- **Data Preprocessing**: Uses Stanza for parsing and preprocessing of medical reports, with a focus on handling complex terminologies.

## Methodology

1. **Data Preprocessing**:
    - Collected and preprocessed 140,000 VAERS reports related to COVID-19 vaccines using Stanza.
    - Created a standard list of symptoms from the VAERS Symptoms table, focusing on the most commonly reported adverse events.
    
2. **Model Training**:
    - Trained a BioBERT model fine-tuned on clinical and biomedical embeddings for Named Entity Recognition (NER) tasks.
    - Separated the dataset into three types: Test, Treatment, and Problem, training individual models for each category.
    
3. **Optimization**:
    - Fine-tuned loss functions (cross-entropy loss) and embeddings to improve model performance.
    - Increased the F1 score by 18% and reduced model loss to 0.0776.
    
4. **Evaluation**:
    - Achieved 97.9% accuracy and an F1 score of 0.96 in symptom extraction tasks.

## Experimental Results

| Model Type      | Eval Loss | Accuracy | Precision | Recall | F1 Score |
|-----------------|-----------|----------|-----------|--------|----------|
| Test            | 0.0776    | 97.9%    | 96.32%    | 96.35% | 96.34%   |
| Treatment       | 0.0205    | 99.5%    | 99.5%     | 87.3%  | 89.6%    |
| Problem         | 0.1272    | 96.5%    | 96.27%    | 96.27% | 96.27%   |

## Dataset

The dataset consists of 140,000 reports from the VAERS database, specifically focusing on the following COVID-19 vaccine variants:

- Pfizer-BioNTech
- Moderna
- Janssen
- Novavax
- Bivalent Variants (Pfizer and Moderna)

More information about the VAERS dataset can be found [here](https://vaers.hhs.gov/about.html).

## Requirements

- Python 3.7+
- Hugging Face Transformers
- Stanza
- PyTorch
- BioBERT

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/vaccine-adverse-events-ner.git
cd vaccine-adverse-events-ner
```

2. Preprocess the data:

```bash
python preprocess.py --data_path path_to_vaers_data
```

3. Train the model:

```bash
python train.py --model_type biobert --epochs 4 --batch_size 16
```

4. Evaluate the model:

```bash
python evaluate.py --model_path saved_model_path
```

## Acknowledgements

This project was made possible using the following resources:
- [BioBERT](https://github.com/dmis-lab/biobert)
- [Stanza](https://stanfordnlp.github.io/stanza/)
- VAERS Database: [vaers.hhs.gov](https://vaers.hhs.gov/about.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- BioBERT: Domain-specific language model pretraining for biomedical NLP.
- Stanza: A Python NLP toolkit for many human languages.
- VAERS Database: The Vaccine Adverse Event Reporting System.

---