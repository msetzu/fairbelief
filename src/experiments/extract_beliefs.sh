python extract_beliefs.py --dataset honest --subset en_binary --model distilroberta-base --dump_file beliefs/honest_en_binary_robertasmall.jsonl;    
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model distilroberta-base --dump_file beliefs/honest_en_queer_nonqueer_robertasmall.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model roberta-base --dump_file beliefs/honest_en_binary_roberta.jsonl;    
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model roberta-base --dump_file beliefs/honest_en_queer_nonqueer_roberta.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model roberta-large --dump_file beliefs/honest_en_binary_robertalarge.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model roberta-large --dump_file beliefs/honest_en_queer_nonqueer_robertalarge.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model gpt2 --dump_file beliefs/honest_en_binary_gpt2.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model gpt2 --dump_file beliefs/honest_en_queer_nonqueer_gpt2.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model gpt2-medium --dump_file beliefs/honest_en_binary_gpt2medium.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model gpt2-medium --dump_file beliefs/honest_en_queer_nonqueer_gpt2medium.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model gpt2-large --dump_file beliefs/honest_en_binary_gpt2large.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model gpt2-large --dump_file beliefs/honest_en_queer_nonqueer_gpt2large.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model gpt2-rstrip --dump_file beliefs/honest_en_binary_gpt2-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model gpt2-rstrip --dump_file beliefs/honest_en_queer_nonqueer_gpt2-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model gpt2-medium-rstrip --dump_file beliefs/honest_en_binary_gpt2medium-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model gpt2-medium-rstrip --dump_file beliefs/honest_en_queer_nonqueer_gpt2medium-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model gpt2-large-rstrip --dump_file beliefs/honest_en_binary_gpt2large-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model gpt2-large-rstrip --dump_file beliefs/honest_en_queer_nonqueer_gpt2large-rstrip.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model distilbert-base-uncased --dump_file beliefs/honest_en_binary_distilbert-base-uncased.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model distilbert-base-uncased --dump_file beliefs/honest_en_queer_nonqueer_distilbert-base-uncased.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model bert-base-uncased --dump_file beliefs/honest_en_binary_bert-base-uncased.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bert-base-uncased --dump_file beliefs/honest_en_queer_nonqueer_bert-base-uncased.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model bert-large-uncased --dump_file beliefs/honest_en_binary_bert-large-uncased.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bert-large-uncased --dump_file beliefs/honest_en_queer_nonqueer_bert-large-uncased.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model albert-base-v2 --dump_file beliefs/honest_en_binary_albert.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model albert-base-v2 --dump_file beliefs/honest_en_queer_nonqueer_albert.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model albert-large-v2 --dump_file beliefs/honest_en_binary_albertlarge.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model albert-large-v2 --dump_file beliefs/honest_en_queer_nonqueer_albertlarge.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model albert-xlarge-v2 --dump_file beliefs/honest_en_binary_albertxlarge.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model albert-xlarge-v2 --dump_file beliefs/honest_en_queer_nonqueer_albertxlarge.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model microsoft/deberta-v3-small --dump_file beliefs/honest_en_binary_debertasmall.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model microsoft/deberta-v3-small --dump_file beliefs/honest_en_queer_nonqueer_debertasmall.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model microsoft/deberta-base --dump_file beliefs/honest_en_binary_deberta.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model microsoft/deberta-base --dump_file beliefs/honest_en_queer_nonqueer_deberta.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model microsoft/deberta-v3-large --dump_file beliefs/honest_en_binary_debertalarge.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model microsoft/deberta-v3-large --dump_file beliefs/honest_en_queer_nonqueer_debertalarge.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model lucadiliello/bart-small --dump_file beliefs/honest_en_binary_bartsmall.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model lucadiliello/bart-small --dump_file beliefs/honest_en_queer_nonqueer_bartsmall.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model facebook/bart-base --dump_file beliefs/honest_en_binary_bart.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model facebook/bart-base --dump_file beliefs/honest_en_queer_nonqueer_bart.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model facebook/bart-large --dump_file beliefs/honest_en_binary_bartlarge.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model facebook/bart-large --dump_file beliefs/honest_en_queer_nonqueer_bart_large.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model t5-base --dump_file beliefs/honest_en_binary_t5_base.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model t5-small --dump_file beliefs/honest_en_binary_t5_small.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model t5-large --dump_file beliefs/honest_en_binary_t5_large.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model t5-3b --dump_file beliefs/honest_en_binary_t5_3b.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model t5-11b --dump_file beliefs/honest_en_binary_t5_11b.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model t5-base --dump_file beliefs/honest_en_queer_nonqueer_t5_base.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model t5-small --dump_file beliefs/honest_en_queer_nonqueer_t5_small.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model t5-large --dump_file beliefs/honest_en_queer_nonqueer_t5_large.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model t5-3b --dump_file beliefs/honest_en_queer_nonqueer_t5_3b.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model t5-11b --dump_file beliefs/honest_en_queer_nonqueer_t5_11b.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model google/flan-t5-base --dump_file beliefs/honest_en_binary_flan-t5_base.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model google/flan-t5-small --dump_file beliefs/honest_en_binary_flan-t5_small.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model google/flan-t5-large --dump_file beliefs/honest_en_binary_flan-t5_large.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model google/flan-t5-xl --dump_file beliefs/honest_en_binary_flan-t5_xl.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model google/flan-t5-xxl --dump_file beliefs/honest_en_binary_flan-t5_xxl.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model google/flan-t5-base --dump_file beliefs/honest_en_queer_nonqueer_flan-t5_base.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model google/flan-t5-small --dump_file beliefs/honest_en_queer_nonqueer_flan-t5_small.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model google/flan-t5-large --dump_file beliefs/honest_en_queer_nonqueer_flan-t5_large.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model google/flan-t5-xl --dump_file beliefs/honest_en_queer_nonqueer_flan-t5_xl.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model google/flan-t5-xxl --dump_file beliefs/honest_en_queer_nonqueer_flan-t5_xxl.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-560m --dump_file beliefs/honest_en_binary_bloom-560m.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-560m --dump_file beliefs/honest_en_queer_nonqueer_bloom-560m.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-3b --dump_file beliefs/honest_en_binary_bloom-3b.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-3b --dump_file beliefs/honest_en_queer_nonqueer_bloom-3b.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-1b3 --dump_file beliefs/honest_en_binary_bloom-1b3.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-1b3 --dump_file beliefs/honest_en_queer_nonqueer_bloom-1b3.jsonl;
# python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-7b1 --dump_file beliefs/honest_en_binary_bloom-7b1.jsonl;
# python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-7b1 --dump_file beliefs/honest_en_queer_nonqueer_bloom-7b1.jsonl;
# python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-1b7 --dump_file beliefs/honest_en_binary_bloom-1b7.jsonl;
# python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-1b7 --dump_file beliefs/honest_en_queer_nonqueer_bloom-1b7.jsonl;

python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-560m-rstrip --dump_file beliefs/honest_en_binary_bloom-560m-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-560m-rstrip --dump_file beliefs/honest_en_queer_nonqueer_bloom-560m-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-3b-rstrip --dump_file beliefs/honest_en_binary_bloom-3b-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-3b-rstrip --dump_file beliefs/honest_en_queer_nonqueer_bloom-3b-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_binary --model bigscience/bloom-1b3-rstrip --dump_file beliefs/honest_en_binary_bloom-1b3-rstrip.jsonl;
python extract_beliefs.py --dataset honest --subset en_queer_nonqueer --model bigscience/bloom-1b3-rstrip --dump_file beliefs/honest_en_queer_nonqueer_bloom-1b3-rstrip.jsonl;