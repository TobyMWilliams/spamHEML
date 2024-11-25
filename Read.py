import json

class Read:

    #label 1 = spam
    #label 0 = ham

    #this means the field 'label text' can be removed
    @staticmethod
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
                    "label": data.get("label")
                })
        return emails


