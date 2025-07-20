import json

class Read:

    #label 1 = spam
    #label 0 = ham

# method to parse the jsonl file and extract the text
    @staticmethod
    def read_jsonl_text(file_path):
        
        emails =[]
        with open(file_path, 'r') as file:
            for line in file:
                    
                data = json.loads(line)

                emails.append(data.get("text", ""))

          
            return emails
    

    # method to parse the jsonl file and extract the labels 
    @staticmethod
    def read_jsonl_label(file_path):

        labels = [] 
        with open(file_path, 'r') as file:
            for line in file:

                data = json.loads(line)

                labels.append(data.get("label", 1))

                 
        return labels
    



