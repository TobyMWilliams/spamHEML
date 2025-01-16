import json

class Read:

    #label 1 = spam
    #label 0 = ham

    #this means the field 'label text' can be removed
    @staticmethod
    def read_jsonl_text(file_path):
        
        emails =[]
        with open(file_path, 'r') as file:
            for line in file:
                    
                data = json.loads(line)

                emails.append(data.get("text", ""))

          
            return emails
    
    @staticmethod
    def read_jsonl_label(file_path):

        labels = [] 
        with open(file_path, 'r') as file:
            for line in file:

                data = json.loads(line)

                labels.append(data.get("label", 0))

                 
        return labels
    



