**Modeling IL2RA Gene Regulation and Lupus Risk through Transformer-Based Analysis**

Explores fine-tuning a DNABERT-2 model to predicts which alleles of the ILR2A gene, whose lack of expression is a key cause of Lupus, are high risk for Lupus. 
This model will be given 3 key inputs to improve its predictions: 
(1) data about location of transcription factors with Position Weight Matrices, 
(2) data about thermodynamic affinity of the transcription factor by looking at DNA biophysical characterisitics, 
(3) data about DNA openness from the predictions of the Enformer model. 

Through this process, hopefully the tuned gLM will be able to give significant genetic insights to which alleles are high risk for Lupus. 




Installation Instructions: 
1. To install dependencies run "pip install -r requirements.txt"
2. While running Task 1, unzip folder "chip_seq"
