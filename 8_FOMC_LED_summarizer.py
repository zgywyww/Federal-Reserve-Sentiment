# Example usage
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
import pandas as pd
import logging

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs available, using DataParallel.")
    multi_gpu = True
else:
    print("Single GPU or CPU available.")
    multi_gpu = False


# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load LED model for summarization and move it to the specified device (GPU if available)
led_model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384').to(device)
led_tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')

def split_text(text, max_chunk_size=1024):
    tokens = bart_tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_chunk_size):
        chunk = bart_tokenizer.convert_tokens_to_string(tokens[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

def summarize_sentence_led(text):
    paragraphs = text.split('\n\n')
    print('====Start===')
    summaries = []
    for paragraph in paragraphs:
        text_list = sent_tokenize(paragraph)
        cur_sentence = ''
        # Predefined settings
        max_input_length = 8000
        max_summary_length = 800
        min_summary_length = 40
        length_penalty = 2.0
        num_beams = 4
        for ii, chunk in enumerate(text_list):
            if ii % 1000 == 0:
                print(f'Progress: {ii}/{len(text_list)}')
            if len(cur_sentence + chunk) < max_input_length - 100 and ii < len(text_list) - 1:
                cur_sentence += chunk + ' '
            else:
                inputs = led_tokenizer(cur_sentence, return_tensors="pt", max_length=max_input_length, truncation=True)
                input_ids = inputs.input_ids.to('cuda')
                attention_mask = inputs.attention_mask.to('cuda')

                # Generate summary
                summary_ids = led_model.generate(input_ids, attention_mask=attention_mask, max_length=max_summary_length, min_length=min_summary_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)

                # Decode summary
                summary = led_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
                cur_sentence = chunk
        summaries.append('\n\n')  # Add a separator for each paragraph summary

    summaries_text = ' '.join(summaries)
    print(summaries_text)
    if len(summaries_text) <= 1024:
        print('===Finished===')
        return summaries_text
    else:
        return summarize_sentence_led(summaries_text)
    
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
start = 0
long_sentence = pd.read_csv('tokenized_sentence.csv')
temp_df = long_sentence.iloc[start:start+5,:]
temp_df['summary'] = temp_df['tokenized'].progress_apply(summarize_sentence_led)
temp_df.to_csv(f'led_summarize.csv')