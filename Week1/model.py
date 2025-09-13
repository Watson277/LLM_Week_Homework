from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = r"C:\\Users\Steven Watson\\.cache\\modelscope\\hub\\models\\langboat\\mengzi-t5-base"

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)  # 移动模型到 GPU

# 测试
inputs = tokenizer("中国的首都在哪", return_tensors="pt").to(device)  # 移动输入到 GPU
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

# 输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))




# from modelscope import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_path = r"C:\Users\Steven Watson\.cache\modelscope\hub\models\langboat\mengzi-t5-base"

# model = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     trust_remote_code=True, 
# )

# tokenizer = AutoTokenizer.from_pretrained(model_path)

# def generate_text(prompt, max_length=50):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs['input_ids'],
#             max_length=max_length,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2,
#             top_p = 0.5,
#             top_k = 50,
#             temperature = 0.7,
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# prompt = "将这句话翻译成英文：我爱自然语言处理"
# result = generate_text(prompt)
# print(result)