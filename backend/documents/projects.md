# Projects

## CERN Dimuon Event Analysis — Personal Project

Fatima completed a machine learning project using CERN Open Data from particle-physics experiments. The project focused on dimuon events and machine learning analysis of CERN particle-physics data.

She built an end-to-end machine learning pipeline using Pandas, NumPy, and scikit-learn on approximately 90,000 dimuon events.

The project covered:

* Regression for invariant mass reconstruction
* Classification of signal versus background
* Anomaly detection for rare events
* SHAP explainability for interpreting model behavior

The analysis identified transverse momentum (pT) and angular features (ΔR, Δη, Δφ) as the main drivers of model predictions.

## Monte Carlo Simulation & Model Validation — Personal Project

Fatima worked on a CERN particle-physics simulation project involving Monte Carlo generation of Z→μμ (Z to muon-muon) events.

She built a 100,000-event Monte Carlo generator using NumPy and SciPy and evaluated machine learning models on real and simulated CERN data.

The project included:

* Monte Carlo event generation
* Regression, classification, and anomaly-detection experiments
* Cross-domain benchmarking
* SHAP-based model interpretability

The analysis identified a strong domain shift: models trained on detector data learned experiment-specific effects and failed to generalize to idealized simulations.

## Fashion MNIST Image Classification — University Project

Built and trained an Artificial Neural Network (ANN) to classify Fashion MNIST clothing images using Python and TensorFlow.

Preprocessed and normalized more than 60,000 images and achieved 88% test accuracy across 10 clothing categories.

## Flower Image Classification — Personal Project

Built a Convolutional Neural Network (CNN) using TensorFlow to classify flower images into different categories, applying deep learning techniques for image classification.

## Customer Segmentation & Business Insights — Personal Project

Used K-Means clustering with scikit-learn and Pandas to segment customers for targeted marketing, combined with data visualization to communicate business insights.

## Distributed Café Management System — University Project

Built a distributed management system for a café franchise network, enabling centralized tracking of orders, inventory, and branch performance using Flask, gRPC, SQL, and Docker.

The project included:

* ML-based analytics to forecast and compare branch performance
* REST API development
* Microservice architecture
* Support for multi-location operations

## AskFatima — Personal RAG Project (ongoing)

AskFatima is Fatima's current and ongoing personal project. She is currently developing a Retrieval-Augmented Generation (RAG) chatbot that answers questions about her background, CV, skills, projects, education, and experience.

The system retrieves information from Fatima's real documents and generates grounded answers based only on the available information rather than generating unsupported facts.

The project involves comparing multiple retrieval strategies:

* Dense embedding retrieval
* BM25 keyword search
* Hybrid retrieval and fusion
* Cross-encoder reranking

Fatima built an evaluation pipeline to measure retrieval accuracy and ranking performance.

The project also includes:

* Guardrails against prompt injection
* Guardrails for off-topic questions
* Citation and source tracing
* Retrieved-context inspection
* React frontend
* FastAPI backend
* Gemini for answer generation
