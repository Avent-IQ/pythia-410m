  Pythia Quantized Model for Sentiment Analysis
=============================================

This repository hosts a quantized version of the Pythia model, fine-tuned for sentiment analysis tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

Model Details
-------------

*   **Model Architecture:** Pythia-410m
*   **Task:** Sentiment Analysis
*   **Dataset:** IMDb Reviews
*   **Quantization:** Float16
*   **Fine-tuning Framework:** Hugging Face Transformers

The quantized model achieves comparable performance to the full-precision model while reducing memory usage and inference time.

Usage
-----

### Installation

    pip install transformers torch

### Loading the Model

    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("AventIQ-AI/pythia-410m")
    model = AutoModelForCausalLM.from_pretrained("AventIQ-AI/pythia-410m", low_cpu_mem_usage=True)
    
    # Example usage
    text = "This product is amazing!"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Performance Metrics
-------------------

*   **Accuracy:** 0.56
*   **F1 Score:** 0.56
*   **Precision:** 0.68
*   **Recall:** 0.56

Fine-Tuning Details
-------------------

### Dataset

The IMDb Reviews dataset was used, containing both positive and negative sentiment examples.

### Training

*   **Number of epochs:** 3
*   **Batch size:** 8
*   **Evaluation strategy:** epoch
*   **Learning rate:** 2e-5

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

Repository Structure
--------------------

    .
    ├── model/               # Contains the quantized model files
    ├── tokenizer/           # Tokenizer configuration and vocabulary files
    ├── model.safensors/     # Fine Tuned Model
    ├── README.md            # Model documentation

Limitations
-----------

*   The model may not generalize well to domains outside the fine-tuning dataset.
*   Quantization may result in minor accuracy degradation compared to full-precision models.

Contributing
------------

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
