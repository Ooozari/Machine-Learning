from transformers import GPT2LMHeadModel,GPT2Tokenizer

tokenizer=GPT2Tokenizer.from_pretrained("gpt2-large")
model=GPT2LMHeadModel.from_pretrained("gpt2-large",pad_token_id=tokenizer.eos_token_id)


sentence='No code needs to adapt to Specialists'
input_ids=tokenizer.encode(sentence,return_tensors='pt')

for i in input_ids[0]:
   print(f"the word is{tokenizer.decode(i)}----------> id:{i}")


output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2)
print(tokenizer.decode(output[0], skip_special_tokens=True))