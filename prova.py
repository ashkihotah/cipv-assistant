

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "dbmdz/bert-base-italian-xxl-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model = model.to(device)
model.eval()  # Set model to evaluation mode since we are not training it