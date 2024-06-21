import os

from icl.icl_acronym import ICLAcronymCaller
from dotenv import load_dotenv

if __name__ == "__main__":
    prompt_model = "gpt-4-1106-preview"
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    # Usage Example
    instruction_onestop = "Given an abstract of a paper, generate a title and an abbreviation for it.  The title should capture the " \
                          "main idea of the paper. The abbreviation works as a shorthand name for the title. The abbreviation " \
                          "is usually composed of characters from the title.  The abbreviation should look like a real word " \
                          "and be easy to remember. One example is as follows:  "
    instruction_onestop_sd = "Given an abstract of a paper, generate an abbreviation and a title for it.  " \
                             "The abbreviation works as a shorthand name for the title. The title should capture the " \
                             "main idea of the paper. The abbreviation is usually composed of characters from the " \
                             "title.  The abbreviation should look like a real word and be easy to remember. One " \
                             "example is as follows: "
    instruction_onestop_char = "Given an abstract of a paper, generate a title and an abbreviation for it.  The title " \
                               "should capture the main idea of the paper. The abbreviation works as a shorthand name " \
                               "for the title. The abbreviation is usually composed of characters from the title.  " \
                               "The abbreviation should look like a real word and be easy to remember. Besides, " \
                               "the abbreviation should be given character by character. One example is as follows:  "
    instruction_only_title = "Given an abstract of a paper, generate a title for it.  The title should capture the " \
                             "main idea of the paper.  One example is as follows:"
    instruction_only_abbreviation_with_abstract_and_generated_title = "Given an abstract of a paper and its title, generate an abbreviation for it.  " \
                                                                      "The abbreviation works as a shorthand name for the title. The abbreviation " \
                                                                      "is usually composed of characters from the title.  The abbreviation should look like a real word " \
                                                                      "and is easy to remember. One example is as follows:  "
    mode = "onestop"
    demonstration_file_path = 'data/icl_demos.jsonl'
    input_file_path = 'data/single_word_with_replacement_test.jsonl'
    output_file_path = "results-icl/icl_{}_{}.jsonl".format(prompt_model, mode)
    d_label = "Title"
    a_label = "Abbreviation"

    processor = ICLAcronymCaller(api_key=api_key, model_name=prompt_model, description_label=d_label,
                                 abbreviation_label=a_label)
    demonstrations = processor.read_demonstrations(demonstration_file_path)
    processed_data = processor.process_jsonl_file(input_file_path,
                                                  output_file_path,
                                                  instruction_onestop,
                                                  instruction_onestop_sd,
                                                  instruction_onestop_char,
                                                  instruction_only_title,
                                                  instruction_only_abbreviation_with_abstract_and_generated_title,
                                                  demonstrations,
                                                  mode
                                                  )
