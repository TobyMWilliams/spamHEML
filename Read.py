import json

class Read:

    #label 1 = spam
    #label 0 = ham

    #this means the field 'label text' can be removed
    def read_jsonl(file_path):
        
        emails =[]
        labels = [] 
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                emails.append({
                    "msg_id": data.get("message_id"),
                    "text" : data.get("text"),
                    "subject": data.get("subject"),
                    "message": data.get("message"),
                    "date": data.get("date"),
                    "label_text": data.get("label_text"),
                    "label": data.get("label")
                })
        return emails



if __name__ == "__main__":
    emails = []
    file_path = "data/train.jsonl"

    emails = Read.read_jsonl(file_path)
    print("first 5 elems of train.jsonl")
    for i, email in enumerate(emails[:5]):
        print (f"\n -- Email {i + 1} ---")
        for key, value in email.items():
            print(f"{key}: {value}")
