import json

def convert(input_dir1, input_dir2, output_dir):
    # Load the two JSON files
    with open(input_dir1, 'r') as f1, open(input_dir2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Initialize a list to hold the DPO-formatted data
    dpo_data = []
    
    # Iterate through the keys in the first file
    for key in data1:
        # Check if the key exists in the second file
        if key in data2:
            # Compare the 'reward' values
            if data1[key]['reward'] != data2[key]['reward']:
                # If 'reward' values are different, extract the instruction and content
                instruction = data1[key]['traj_by_line'][3]
                content1_lines = data1[key]['traj_by_line'][6:]
                content1 = '\n'.join(content1_lines)
                content2_lines = data2[key]['traj_by_line'][6:]
                content2 = '\n'.join(content2_lines)
                 # Create a dictionary in the Alpaca format
                dpo_dict = {
                    "instruction": instruction,
                    "input": "", 
                    "chosen": content1 if data1[key]['reward'] else content2,  # Assuming there is no additional input required
                    "rejected": content2 if data1[key]['reward'] else content1
                }
                # Print or process the extracted information
                print(f"Instruction: {instruction}")
                print(f"Content from file 1: {content1}")
                print(f"Content from file 2: {content2}")
                # Append the dictionary to the list
                dpo_data.append(dpo_dict)
                
    # Write the Alpaca-formatted data to the output JSON file
    with open(output_dir, 'w') as output_file:
        json.dump(dpo_data, output_file, indent=4)
            
if __name__ == '__main__':
    data_path = 'trajs/'
    original_file1 = data_path + 'hotpotqa_dev_0_500_llama_models_lora_llama-2-7b-d_1000_0.7_2023-09-19-00-49-36.json'
    original_file2 = data_path + 'hotpotqa_dev_0_500_llama_models_lora_llama-2-13b-d_1000_0.7_2023-09-18-13-40-08.json'
    output_file_path = data_path + 'dpo_format.json'
    convert(original_file1, original_file2, output_file_path)
    print(f"Data converted to DPO format and saved to {output_file_path}")