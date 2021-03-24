# EDGE-exemplars
Code for the paper [Controlling Dialogue Generation with Semantic Exemplars](https://arxiv.org/abs/2008.09075) (NAACL 2021) A semantic exemplar based retrieve-refine approach for dialogue response generation

## Introduction
This work lets you control dialogue response genration based on exemplar responses. Our model EDGE uses the semantic structure of an exemplar response, instead of the tokens of the exemplar response, to guide the generation.

For a novel dialogue context, we retrieve a human-written response exemplar and represent it using its semantic frames. We then incorporate the dialogue context and the semantic frames of the response exemplars in a GPT-2 based conditional language model, thereby combining the benefits of fluency of language models and the semantic guidance of the exemplar responses structured with rich linguistic knowledge. 

Semantic frames capture the meaning of the exemplars rather than their surface forms, while preserving the semantic structure of the exemplars better than current token-based retrieve-refine approaches. EDGE generates exemplar-conditioned responses that are coherent, context-specific, and adherent to underlying exemplar intents and their high-level goals. 


## Paper Abstract

> Dialogue systems pretrained with large language models generate locally coherent responses, but lack fine-grained control over responses necessary to achieve specific goals. A promising method to control response generation isexemplar-based generation, in which models edit exemplar responses that are retrieved from training data, or hand-written to strategically address discourse-level goals, to fit new dialogue contexts. However, current exemplar-based approaches often excessively copy words from the exemplar responses, leading to incoherent replies. We present an Exemplar-based Dialogue Generation model, EDGE, that uses the semantic frames present in exemplar responses to guide response generation. We show that controlling dialogue generation based on the semantic frames of exemplars improves the coherence of generated responses, while preserving semantic meaning and conversation goals present in exemplar responses.

## Model diagram and input representation
![Model figure](https://github.com/prakharguptaz/EDGE-exemplars/blob/main/input-figure.png?raw=true)

The input representation of our proposed approach. During training, EDGE conditions on the dialogue context and a noisy version of the ground truth response semantic frames to generate the ground truth response. During inference, we feed the context and the semantic frames from the response exemplars to generate a response.

## Data Processing
The code and sample files for data processing along with a readme are present in the folder `data_processing`

## Model Training

Requirements: Code for the models used transformers==2.4.0 and torch==1.4.0 (later versions of torch should also work)

To train the *EDGE* model, run the train_robust.py file

```console
python train_robust.py --dataset_path $LOCATION_OF_JSON_DATA`
```

To generate responses for the test json, you can euther run the script run_generate.py, or its multiprocessing version run_generate_batch.py
```console
python run_generate.py --model_checkpoint $TRAINED_MODEL_FOLDER --dataset_path $LOCATION_OF_JSON_DATA`
```
```console
python run_generate_batch.py --model_checkpoint $TRAINED_MODEL_FOLDER --dataset_path $LOCATION_OF_JSON_DATA`
```