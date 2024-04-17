import json

def convert(input_dir, output_dir):
    # Define the path to the input JSON file
    input_file_path = input_dir
    # Define the path to the output JSON file
    output_file_path = output_dir

    # Read the input JSON file
    with open(input_file_path, 'r') as input_file:
        data = json.load(input_file)

    # Initialize a list to hold the Alpaca-formatted data
    alpaca_data = []

    # Iterate over the items in the input JSON data
    for key, value in data.items():
        # Extract the instruction and output from the 'traj_by_line' field
        instruction = value['traj_by_line'][3]  # Assuming the instruction is always on the 4th line
        output_lines = value['traj_by_line'][6:]  
        output = '\n'.join(output_lines)
        # Create a dictionary in the Alpaca format
        alpaca_dict = {
            "instruction": instruction,
            "input": "",  # Assuming there is no additional input required
            "output": output
        }
        # Append the dictionary to the list
        alpaca_data.append(alpaca_dict)

    # Write the Alpaca-formatted data to the output JSON file
    with open(output_dir, 'w') as output_file:
        json.dump(alpaca_data, output_file, indent=4)
        
if __name__ == '__main__':
    data_path = 'trajs/'
    original_file = data_path + 'hotpotqa_dev_0_5_llama__home_lhmeng_rlproj_llm_rl_agent_model_SPIN_outputs_iter0-ckpt_0.7_2024-04-17-14-43-24.json'
    output_file_path = data_path + 'alpaca_format.json'
    convert(original_file, output_file_path)
    print(f"Data converted to Alpaca format and saved to {output_file_path}")