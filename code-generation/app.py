import gradio as gr
from task2_code_generation_finetune import prepare_model, generate_code

if __name__ == "__main__":
    model, tokenizer = prepare_model(model_name="final_spoc_distilgpt2_lora")
    
    gr.Interface(
        fn=lambda pseudo: generate_code(model, tokenizer, pseudo),
        inputs="text",
        outputs="code",
        title="SPoC Pseudo→C++ Code Generator",
        description="Fine-tuned model on SPoC dataset"
    ).launch(share=False)

