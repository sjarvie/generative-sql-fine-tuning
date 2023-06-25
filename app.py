import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"stjarvie/bloom-1b7-sql-generation"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


def generate_prompt_inference(question: str, schema: str) -> str:
  prompt = f"### Question:\n{question}\n\n### Table Schema:\n{schema}\n\n### SQL Query:\n "
  return prompt


def make_inference(question, schema):
    batch = tokenizer(
        generate_prompt_inference(question, schema),
        return_tensors="pt",
    )

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=200)

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)


if __name__ == "__main__":
    # make a gradio interface
    import gradio as gr

    gr.Interface(
        make_inference,
        [
            gr.inputs.Textbox(lines=2, label="Question"),
            gr.inputs.Textbox(lines=5, label="Table Schema"),
        ],
        gr.outputs.Textbox(label="Generated Query"),
        title="Generative-SQL-AI",
        description="This is a tool that generates SQL given a question and related table schema.",
    ).launch()