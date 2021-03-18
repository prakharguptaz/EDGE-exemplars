# EDGE-exemplars
Code for the paper - Controlling Dialogue Generation with Semantic Exemplars (NAACL 2021) A semantic exemplar based retrieve-refine approach for dialogue response generation

## Data Processing
The code and sample files for data processing along with a readme are present in the folder `data_processing`

## Model Training
To train the *EDGE* model, run the train_robust.py file
`python train_robust.py --dataset_path $LOCATION_OF_JSON_DATA`

To generate responses for the test json, you can euther run the script run_generate.py, or its multiprocessing version run_generate_batch.py

`python run_generate.py --model_checkpoint $TRAINED_MODEL_FOLDER --dataset_path $LOCATION_OF_JSON_DATA`
`python run_generate_batch.py --model_checkpoint $TRAINED_MODEL_FOLDER --dataset_path $LOCATION_OF_JSON_DATA`
