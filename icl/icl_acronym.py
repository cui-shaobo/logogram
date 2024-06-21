import os

import json

from tqdm import tqdm
from openai import OpenAI


class ICLAcronymCaller:
    def __init__(self, api_key, model_name="gpt-4-1106-preview", description_label="Description",
                 abbreviation_label="Acronym"):
        self.client = OpenAI()
        self.api_key = api_key
        self.model_name = model_name
        self.description_label = description_label
        self.abbreviation_label = abbreviation_label

        OpenAI.api_key = self.api_key

    @staticmethod
    def read_demonstrations(file_path):
        demonstrations = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                demonstrations.append(data)
        return demonstrations

    # This method is for creating prompts for generating title and abbreviation in an onestop manner.
    def create_prompt_with_demonstrations_onestop(self, instruction, demonstrations, abstract):
        prompt = f"{instruction}\n"
        for demo in demonstrations:
            prompt += f"Abstract: {demo['Abstract']} {self.description_label}: {demo['Description']} | {self.abbreviation_label}: {demo["Acronym"]}\n"
        prompt += f" Given the following abstract, generated a title and an abbreviation for it. The title and abbreviation are separated by |. " \
                  f"Abstract: {abstract} \n "
        return prompt

    # This method is for creating prompts for generating title and abbreviation in an onestop manner,
    # but the abbreviation is given character by character.
    def create_prompt_with_demonstrations_onestop_char(self, instruction, demonstrations, abstract):
        prompt = f"{instruction}\n"
        for demo in demonstrations:
            char_abbreviation_demo = '  '.join(list(demo["Acronym"]))
            prompt += f"Abstract: {demo['Abstract']} {self.description_label}: {demo['Description']} | {self.abbreviation_label}: {char_abbreviation_demo}\n"
        prompt += f" Given the following abstract, generated a title and an abbreviation for it. The abbreviation " \
                  f"should be given character by character. The title and abbreviation are separated by |. \n Abstract: {abstract} \n "
        return prompt

    # This method is for creating prompts for generating abbreviation(first) and title(then) in an onestop manner.
    def create_prompt_with_demonstrations_onestop_sd(self, instruction, demonstrations, abstract):
        prompt = f"{instruction}\n"
        for demo in demonstrations:
            prompt += f"Abstract: {demo['Abstract']} {self.abbreviation_label}: {demo["Acronym"]} | {self.description_label}: {demo['Description']}. \n"
        prompt += f" Given the following abstract, generated an abbreviation and a title for it. The abbreviation and the title are separated by |. " \
                  f"Abstract: {abstract} \n "
        return prompt

    # This method is for creating prompts for generating only title with abstract.
    def create_prompt_with_demonstrations_only_title(self, instruction, demonstrations, abstract):
        prompt = f"{instruction}\n"
        for demo in demonstrations:
            prompt += f"Abstract: {demo['Abstract']} {self.description_label}: {demo['Description']}. \n"
        prompt += f"Given the following abstract, generated a title for it. \n" \
                  f"Abstract: {abstract} \n "
        # print('Prompt for title in pipeline: {}'.format(prompt))
        return prompt

    # This method is for creating prompts for generating only abbreviation with {abstract, generated_title}.
    def create_prompt_with_demonstrations_only_abbreviation_with_abstract_and_title(self, instruction, demonstrations,
                                                                                    abstract, generated_title):
        prompt = f"{instruction}\n"
        for demo in demonstrations:
            prompt += f"Abstract: {demo['Abstract']} {self.description_label}: {demo['Description']} \n{self.abbreviation_label}: {demo["Acronym"]}\n"
        prompt += f"Given the following abstract and its title, generate an abbreviation for the abstract. \n" \
                  f"Abstract: {abstract} {self.description_label}: {generated_title} \n"
        # print('Prompt for abbreviation in pipeline: {}'.format(prompt))
        return prompt

    def generate_only_title(self, instruction, demonstrations, abstract):
        prompt = self.create_prompt_with_demonstrations_only_title(instruction, demonstrations, abstract)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=30
        )
        title = response.choices[0].message.content
        parts = title.split(':')
        title_string = parts[-1].strip() if len(parts) > 1 else parts[0].strip()
        if title_string.endswith('"'):
            title_string = title_string[:-1]
        else:
            title_string = title_string

        if title_string.startswith('"'):
            title_string = title_string[1:]
        else:
            title_string = title_string

        return title_string

    def generate_acronym_with_abstract_and_generated_title(self, instruction, demonstrations, abstract,
                                                           generated_title):
        prompt = self.create_prompt_with_demonstrations_only_abbreviation_with_abstract_and_title(instruction,
                                                                                                  demonstrations,
                                                                                                  abstract,
                                                                                                  generated_title)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=30
        )
        abbreviation = response.choices[0].message.content
        parts = abbreviation.split(':')
        abbreviation_string = parts[-1].strip() if len(parts) > 1 else parts[0].strip()

        return abbreviation_string

    def generate_title_and_acronym_onestop_char(self, instruction, demonstrations, abstract, max_attempts=5):
        for attempt in range(max_attempts):
            try:
                prompt = self.create_prompt_with_demonstrations_onestop_char(instruction, demonstrations, abstract)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=30
                )
                output = response.choices[0].message.content

                # Splitting the output into description and acronym
                parts = output.split('|')
                description_string = parts[0].strip() if len(parts) > 0 else ""
                acronym_string = parts[1].strip() if len(parts) > 1 else ""

                description_parts = description_string.split(":")
                acronym_parts = acronym_string.split(":")

                description = description_parts[1].strip() if len(description_parts) > 1 else description_parts[
                    0].strip()
                description = description.rstrip()
                acronym = acronym_parts[1].strip() if len(acronym_parts) > 1 else acronym_parts[0].strip()
                acronym = acronym.replace(" ", "")

                # Ensuring acronym is not empty
                assert acronym != ""
                return description, acronym

            except AssertionError:
                print(f"Output: {output}")
                print(f'Attempt {attempt + 1} failed, retrying...')
                continue
            except Exception as e:
                print(f'An error occurred: {e}')
                break

        raise ValueError('Failed to generate valid output after maximum attempts.')

    def generate_title_and_acronym_onestop(self, instruction, demonstrations, abstract):
        prompt = self.create_prompt_with_demonstrations_onestop(instruction, demonstrations, abstract)
        # print("prompt: {}".format(prompt))
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=40
        )
        output = response.choices[0].message.content
        # print(f"output: {output}")
        # Splitting the output into description and acronym
        parts = output.split('|')
        description_string = parts[0].strip() if len(parts) > 0 else ""
        acronym_string = parts[1].strip() if len(parts) > 1 else ""
        description_parts = description_string.split(":")
        acronym_parts = acronym_string.split(":")
        try:
            # print('output: {}'.format(output))
            description = description_parts[1].strip() if len(description_parts) > 1 else description_parts[0].strip()
            acronym = acronym_parts[1].strip() if len(acronym_parts) > 1 else acronym_parts[0].strip()
        except:
            print('output: {}'.format(output))
            raise ValueError('strange output!!!!')

        # print(f"description: {description}, acronym: {acronym}")

        return description, acronym

    def generate_acronym_and_title_onestop_sd(self, instruction, demonstrations, abstract):
        prompt = self.create_prompt_with_demonstrations_onestop_sd(instruction, demonstrations, abstract)
        print("prompt: {}".format(prompt))
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=30
        )
        output = response.choices[0].message.content
        # print(f"output: {output}")
        # Splitting the output into description and acronym
        parts = output.split('|')
        try:
            acronym_string = parts[0].strip() if len(parts) > 1 else ""
            description_string = parts[1].strip() if len(parts) > 0 else ""
            description_parts = description_string.split(":")
            acronym_parts = acronym_string.split(":")
            # print('output: {}'.format(output))
            description = description_parts[1].strip() if len(description_parts) > 1 else description_parts[0].strip()
            acronym = acronym_parts[1].strip() if len(acronym_parts) > 1 else acronym_parts[0].strip()
        except:
            print('output: {}'.format(output))
            raise ValueError('strange output!!!!')

        # print(f"description: {description}, acronym: {acronym}")

        return acronym, description

    def process_jsonl_file(self, input_file_path, output_file_path, instruction_onestop, instruction_onestop_sd,
                           instruction_onestop_char, instruction_only_title,
                           instruction_only_abbreviation_with_abstract_and_generated_title,
                           demonstrations, mode):
        assert mode in ['onestop', 'onestop_sd', 'onestop_char', 'pipeline']

        processed_abstracts = set()

        # Check if output file exists and read processed abstracts
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as output_file:
                for line in output_file:
                    processed_data = json.loads(line)
                    processed_abstracts.add(processed_data.get("Abstract", ""))

        with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
            lines = input_file.readlines()
            for line in tqdm(lines, desc="Processing Abstracts"):
                data = json.loads(line)
                abstract = data.get("Abstract", "")
                ground_truth_description = data.get("Description", "")
                ground_truth_acronym = data.get("Acronym", "")

                if abstract and abstract not in processed_abstracts:
                    if mode == 'onestop':
                        generated_title, acronym = self.generate_title_and_acronym_onestop(instruction_onestop,
                                                                                           demonstrations,
                                                                                           abstract)
                    elif mode == 'onestop_sd':
                        acronym, generated_title = self.generate_acronym_and_title_onestop_sd(instruction_onestop_sd,
                                                                                              demonstrations,
                                                                                              abstract)
                    elif mode == 'onestop_char':
                        generated_title, acronym = self.generate_title_and_acronym_onestop_char(
                            instruction_onestop_char,
                            demonstrations,
                            abstract)
                    elif mode == 'pipeline':
                        generated_title = self.generate_only_title(instruction_only_title, demonstrations, abstract)
                        print(f'generated title of pipeline: {generated_title}')
                        acronym = self.generate_acronym_with_abstract_and_generated_title(
                            instruction_only_abbreviation_with_abstract_and_generated_title, demonstrations, abstract,
                            generated_title)
                        print(f'generated acronym of pipeline: {acronym}')

                    print(f"generated_title: {generated_title}, acronym: {acronym}")
                    result = {
                        "Mode": mode,
                        "Type": data.get("Type", ""),
                        "Year": data.get("Year", ""),
                        "Area": data.get("Area", ""),
                        "Where": data.get("Where", ""),
                        "Generated Title": generated_title,
                        "Generated Abbreviation": acronym,
                        "Ground Truth Title": ground_truth_description,
                        "Ground Truth Abbreviation": ground_truth_acronym,
                        "Abstract": abstract,
                    }

                    # Write the result to the output file
                    output_file.write(json.dumps(result) + '\n')
                    output_file.flush()  # Haha, flush method.


def remove_empty_key_lines(input_jsonl_file, output_jsonl_file, key):
    with open(input_jsonl_file, 'r') as infile, open(output_jsonl_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                # Check if the key exists and its value is not an empty string
                if key in data and data[key] != '':
                    outfile.write(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
