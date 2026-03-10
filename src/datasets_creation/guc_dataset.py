import pandas as pd

dataset_1 = pd.read_csv('files/corpus_spa_guc.csv')

dataset_2 = pd.read_csv('files/diccionario_wayuunaiki.csv', sep='|')
dataset_2 = dataset_2.reindex(columns=['guc', 'spa'])

with open('files/data/release/v2021-08-07/guc-spa/train.trg', newline='\n') as f:
    spa = f.readlines()
spa = [line.strip() for line in spa]

with open('files/data/release/v2021-08-07/guc-spa/train.src', newline='\n', encoding='utf-8') as f:
    guc = f.readlines()
guc = [line.strip() for line in guc]
spa_guc = pd.DataFrame({'spa': spa, 'guc': guc})

guc_spa_dataset = pd.concat([dataset_1, dataset_2, spa_guc], ignore_index=True)

guc_spa_dataset.to_csv('spa_guc_dataset.csv', index=False, sep="|")

word_counts = guc_spa_dataset['spa'].apply(lambda x: len(x.split()))

total_words = word_counts.sum()

print(f'Total number of words in the "spa" column: {total_words}')

num_records = guc_spa_dataset.shape[0]

print(f'Number of records in the DataFrame: {num_records}')

unique_words = set()

for text in guc_spa_dataset['spa']:
    unique_words.update(text.split())

num_unique_words = len(unique_words)

print(f'Number of unique words in the "spa" column: {num_unique_words}')
