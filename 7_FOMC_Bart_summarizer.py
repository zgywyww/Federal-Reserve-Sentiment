# Example usage
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import logging


# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs available, using DataParallel.")
    multi_gpu = True
else:
    print("Single GPU or CPU available.")
    multi_gpu = False
    
# Load BART model for summarization and move it to the specified device (GPU if available)
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def split_text(text, max_chunk_size=1024):
    tokens = bart_tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_chunk_size):
        chunk = bart_tokenizer.convert_tokens_to_string(tokens[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

def summarize_sentence(text_list):
    print("Start_summarize")
    summaries = []
    cur_sentence = ''
    for ii, chunk in enumerate(text_list):
        if ii%1000==0:
            print(f"Progress{ii}/{len(text_list)})")
        if len(cur_sentence+chunk) < 1000 and ii < len(text_list)-1:
            cur_sentence = cur_sentence+chunk
        else:
            inputs = bart_tokenizer([cur_sentence], max_length=1024, return_tensors='pt', truncation=True)

            # Move input tensors to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate summary and move it back to CPU for decoding
            summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
            cur_sentence = chunk
    summaries_text = ' '.join(summaries)
    if len(summaries_text) <= 1024:
        return summaries_text
    else:
        return summarize_sentence(summaries)
    
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
start = 0
long_sentence = pd.read_csv('tokenized_sentence.csv')
temp_df = long_sentence.iloc[start:start+5,:]
temp_df['summary'] = temp_df['tokenized'].progress_apply(summarize_sentence)
temp_df.to_csv(f'bart_summarize.csv')