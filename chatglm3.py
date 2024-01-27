from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True).float()
# model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

History=[]
def  glm3Chat(txt):
    global History
    response, history = model.chat(tokenizer, txt, history=History)
    History = history
    return response