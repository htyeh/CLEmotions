#!/usr/local/bin/python3
import re
import random
import time
import json
import googletrans

# https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\u2640-\u2642"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'', text)

num_samples = 500
tgt_lang = 'pt'

start_time = time.time()
# step 1: randomly sample and remove sampled lines from En dataset
with open('full_dataset.tsv') as full_set:
    en_lines = full_set.readlines()
    sampled_lines = random.sample(en_lines, num_samples)
    en_lines = [line for line in en_lines if line not in sampled_lines]
with open('rest_dataset.tsv', 'w') as rest_set:
    for line in en_lines:
        rest_set.write(line)

with open('en_' + tgt_lang + '.tsv', 'w') as src_dataset, open(tgt_lang + '.tsv', 'w') as tgt_dataset:
# step 2: translate texts in sampled_lines
    translator = googletrans.Translator()
    counter = 0 # sleep 10s after every 20 translations
    for line in sampled_lines:
        src_dataset.write(line) # back up sampled en data
        if counter < 40:
            counter += 1
            text, labels, id = line.split('\t')
            try:
                translated_text = translator.translate(text, src='en', dest=tgt_lang).text
                print(text)
                print(translated_text)
            except json.decoder.JSONDecodeError:
                try: # remove emojis
                    translated_text = translator.translate(deEmojify(text), src='en', dest=tgt_lang).text
                    print(text)
                    print(translated_text)
                except json.decoder.JSONDecodeError: # if still containing emojis out of range
                    translated_text = text
                    print('failed to translate:', text)  # this is a problematic sentence
                    print()
            translated_line = translated_text + '\t' + labels + '\t' + id
            tgt_dataset.write(translated_line)
            time.sleep(2)
        else:
            counter = 0
            text, labels, id = line.split('\t')
            try:
                translated_text = translator.translate(text, src='en', dest=tgt_lang).text
                print(text)
                print(translated_text)
            except json.decoder.JSONDecodeError:
                try: # remove emojis
                    translated_text = translator.translate(deEmojify(text), src='en', dest=tgt_lang).text
                    print(text)
                    print(translated_text)
                except json.decoder.JSONDecodeError: # if still containing emojis out of range
                    translated_text = text
                    print('failed to translate:', text)  # this is a problematic sentence
                    print()
            translated_line = translated_text + '\t' + labels + '\t' + id
            tgt_dataset.write(translated_line)
            time.sleep(10)

# print(len(sampled_lines))
# print(len(translated_lines))
# step 3: write original sampled lines with their translations together into an inspection file
# with open(tgt_lang + '_inspect.tsv', 'w') as inspect_file:
#     for i in range(num_samples):
#         inspect_file.write(sampled_lines[i] + translated_lines[i] + '\n')

# step 3: back up sampled en data
# with open('en_' + tgt_lang + '.tsv', 'w') as src_dataset:
#     for line in sampled_lines:
#         src_dataset.write(line)

# step 4: create the translated dataset
# with open(tgt_lang + '.tsv', 'w') as tgt_dataset:
#     for line in translated_lines:
#         tgt_dataset.write(line)

end_time = time.time()
print("Finished in " + str(round((end_time - start_time) / 60, 2)) + " minute(s).")
