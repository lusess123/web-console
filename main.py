from transformers import pipeline; 
pipeline = pipeline('sentiment-analysis')
# result = pipeline('we love you')
# print(result)
while True:
    user_input = input("please input: ")
    if user_input == "exit":
        print("Program terminated.")
        break
    else:
        print("you said: " + user_input)
        result = pipeline(user_input)
        print(result)
        result_str = "I said : {label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
        print(result_str)

