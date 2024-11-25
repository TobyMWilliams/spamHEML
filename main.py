from Read import Read

def main():
    # Specify the path to your JSONL file
    file_path = "train.jsonl"  # Adjust the path if the file is in a different directory
    
    # Call the read_jsonl method from the Read class
    emails = Read.read_jsonl(file_path)
    
    # Print the first 5 emails
    print("First 5 emails of train.jsonl:")
    for i, email in enumerate(emails[:5]):
        print(f"\n -- Email {i + 1} ---")
        for key, value in email.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
