python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path='/mnt/nfs/scratch1/xiangl/family_feud/models/large_outputb_1e_1gu_8' \
    --length=10 \
    --num_samples=300 \
    --temperature=0.69 \
    --input_file='./crowdsource_test.jsonl'
