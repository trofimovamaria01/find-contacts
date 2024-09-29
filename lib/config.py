import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../bert_final_model"
truncation = True
max_length = 512

phone = r"(?:(\+?[7|8|9])([\-\(\)а-яА-Яa-zA-Z_ ]{0,10}))?(\d{3})([\-\(\)а-яА-Яa-zA-Z_ ]{0,10})(\d{3})([\-\(\)а-яА-Яa-zA-Z_ ]{0,10})(\d{2})([\-\(\)а-яА-Яa-zA-Z_ ]{0,10})(\d{2})"
username = r"(@\w{4,32})|(?:(https?:\/?\/)?t(elegram)?\.me\/(\w{5,}))"
vk = r"(http:\/?\/?|https:\/?\/?)?(www.)?(vk\.com|vkontakte\.ru)\/(id\d|[a-zA-Z0-9_.]){2,}"

CONTACT = ";".join([phone, username, vk])
