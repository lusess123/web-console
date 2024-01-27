import torch
from transformers import AutoModel, AutoTokenizer
# GPU设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型与tokenizer
model_name_or_path = 'scutcyr/SoulChat'
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 单轮对话调用模型的chat函数
user_input = "我失恋了，好难受！"
input_text = "用户：" + user_input + "\n心理咨询师："
response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)
print(response)
print(history)
# # 多轮对话调用模型的chat函数
# # 注意：本项目使用"\n用户："和"\n心理咨询师："划分不同轮次的对话历史
# # 注意：user_history比bot_history的长度多1
# user_history = ['你好，老师', '我女朋友跟我分手了，感觉好难受']
# bot_history = ['你好！我是你的个人专属数字辅导员甜心老师，欢迎找我倾诉、谈心，期待帮助到你！']
# # 拼接对话历史
# context = "\n".join([f"用户：{user_history[i]}\n心理咨询师：{bot_history[i]}" for i in range(len(bot_history))])
# input_text = context + "\n用户：" + user_history[-1] + "\n心理咨询师："

# response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)