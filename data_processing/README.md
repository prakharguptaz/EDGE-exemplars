#Data Processing

## Frame extraction
We use the open-sesame model for frame detection https://github.com/swabhs/open-sesame
Other options for frame semantic parsing are (open-source implementation is pending for some):
[Option 2](https://arxiv.org/abs/2101.12175) (LOME)
[Option 3](https://arxiv.org/abs/2010.10998)
[Option 4](https://arxiv.org/abs/2011.13210)

Please follow the steps listed in the open-sesame repo to install the dependencies and then train the model (pretrained model has some know issues). Once the model is trained, please use the following commands to generate frames in CONLL format for sentences in a file

    python -m sesame.targetid --mode predict \
                            --model_name $TARGETID_MODEL_NAME \
                            --raw_input $INPUT_FILE
    python -m sesame.frameid --mode predict \
                           --model_name $FRAMEID_MODEL_NAME \
                           --raw_input logs/$TARGETID_MODEL_NAME/predicted-targets.conll
						   
The above commands will generate a predicted-targets.conll file. We are providing a file "conll_parse.py" which takes the "predicted-targets.conll" file as input and generates a csv with two columns - the input text and the frames for each token of the text. Tokens with no frames are represented by underscores. For input sentences with no frame detected, the file generates a blank row (since the conll file does not contain the corresponding input text). Please put this file in the "sesame" folder of the open-sesame repo. Sample command: 
  
    python -m sesame.conll_parse -o sesame/outs logs/$FRAMEID_MODEL_NAME/predicted-targets.conll 

## Data Preparation for the model

The file `convert_convai_lm_data.py` prepares the data for the model training in the required format. The script takes csv files as input (corresponding to train, test and valid sets). The csv files should contain the context, response, exemplar response and their frames.
A sample file `sample_frame_data.csv` is provided as input, and the first row of the csv file contains the column names (id, context,	response,	retrieved context,	retrieved response (or exemplar response),	response frames,	retrieved response frames). The retrieved response and retrieved response frames columns are not used in training data preparation. A sample output of the script is uploaded with the name `lm_data.json`