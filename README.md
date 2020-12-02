# ProtoQA-GPT2 Baseline

This repo contains the GPT2 baselines in the [ ProtoQA ](https://www.aclweb.org/anthology/2020.emnlp-main.85.pdf) paper. 
- The dataset is here: [https://github.com/iesl/protoqa-data](https://github.com/iesl/protoqa-data)
- The proposed evaluation is here: [https://github.com/iesl/protoqa-evaluator](https://github.com/iesl/protoqa-evaluator)

## Requirments
- Pytorch: 1.4.0
- Huggingface transformer: 2.1.1

Following the install.sh in the repo will create a conda enviroment named protoqa, with corresponding libraries installed. Note: protoqa-evaluater is included. 

## Fine-tuned Model and Inference
- Fine-tuned model can be download here: 
- Generate answers using the fine-tuned GPT2 model (or run generation.sh):
`python run_generation.py \` 
	`--model_type=gpt2 \`
   ` --model_name_or_path='./models/large_outputb_1e_1gu_8' \`
    `--length=10 \`
    `--num_samples=300 \`
    `--temperature=0.69 \`
    `--input_file='./data/dev/crowdsource_dev.jsonl'`
    `--output='./'`
    This will generate ranked_answer.jsonl under the same directionary. 
- Run protoqa-evaluator, for example:
`protoqa_evaluator evaluate --similarity_function exact_match targets.jsonl ranked_answer.jsonl`
- For detail usage of protoqa-evaluator, please refer to https://github.com/iesl/protoqa-evaluator


