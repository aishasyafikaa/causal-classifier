
from UniCausal._datasets.unifiedcre import load_cre_dataset, available_datasets

print('List of available datasets:', available_datasets)

"""
 Example case of loading AltLex and BECAUSE dataset,
 without adding span texts to seq texts, span augmentation or user-provided datasets,
 and load both training and validation datasets.
"""

span_data, seqpair_data, stats = load_cre_dataset(dataset_name=['altlex', 'because', 'ctb', 'semeval2010t8'], do_train_val=True, data_dir='UniCausal/data')

print("=== Span-level Dataset ===")
print(span_data)

print("\n=== Sequence/Pair-level Dataset ===")
print(seqpair_data)

print("\n=== Dataset Stats (min_batch_size, pspan, apair, aseq) ===")
print(stats)


'''dataset_sources_to_show = ['altlex', 'because', 'ctb', 'semeval2010t8'] # esl and pdtb not available
dataset = load_cre_dataset(
    dataset_name=dataset_sources_to_show,
    do_train_val=True,
    data_dir='UniCausal/data'
)

# Span examples
dataset_sources_shown = []
print("SPAN EXAMPLES")
for i in dataset[0]['span_validation']:
    corpus = i['corpus']
    if corpus not in dataset_sources_shown:
        print(i,'\n')
        dataset_sources_shown.append(corpus)

# Seq examples
dataset_sources_shown = []
print("SEQ EXAMPLES")
for i in dataset[1]['seq_validation']:
    corpus = i['corpus']
    if corpus not in dataset_sources_shown:
        print(i,'\n')
        dataset_sources_shown.append(corpus)

# Pair examples
dataset_sources_shown = []
print("PAIR EXAMPLES")
for i in dataset[1]['pair_validation']:
    corpus = i['corpus']
    if corpus not in dataset_sources_shown:
        print(i,'\n')
        dataset_sources_shown.append(corpus)'''
