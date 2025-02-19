use std::path::PathBuf;
use std::io::{self, Write};
use tokenizers::Tokenizer;

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

fn main() {
    // 获取用户输入来选择模型
    let model_type = get_user_choice();

    // 根据选择加载模型
    let (llama, tokenizer) = get_model_params(&model_type);

    if model_type == "story" {
        story_model(llama, tokenizer);
    } else {
        chat_model(llama, tokenizer);
    }


}

fn get_user_choice() -> String {
    println!("Please choose a model (story/chat):");

    let mut choice = String::new();
    io::stdout().flush().unwrap();  // Make sure the prompt is printed before user input
    io::stdin().read_line(&mut choice).unwrap();

    let choice = choice.trim().to_lowercase();

    // 检查输入是否有效
    if choice == "story" || choice == "chat" {
        return choice;
    } else {
        println!("Invalid choice. Defaulting to 'story' model.");
        return "story".to_string();  // Default to 'story' if input is invalid
    }
}

fn get_model_params(name: &str) -> (model::Llama::<f32>, Tokenizer) {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join(name);
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    (llama, tokenizer)
}

// ======== Story Model ========
fn story_model(llama : model::Llama<f32>, tokenizer : Tokenizer) {
    // 在这里可以定义有关故事模型的特定操作
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("{}", input);

    // 调用生成函数
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

// ======== Chat Model ========
fn chat_model(llama : model::Llama<f32>, tokenizer : Tokenizer) {
    println!("Welcome to the chat model! Type 'exit' to quit.");
    while true {
        print!("You: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        if input.trim() == "exit" {
            println!("Goodbye!");
            break;
        }

        let input = create_prompt(input);
        // println!("Prompt: {}", input);
        // std::process::exit(0);
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();

        // 调用生成函数
        let output_ids = llama.generate(
            input_ids,
            100,
            0.8,
            4,
            1.,
        );

        let output = tokenizer.decode(&output_ids, true).unwrap();
        println!("Bot: {}", output);

    }
    // 在这里可以定义有关聊天模型的特定操作
}

fn create_prompt(input :String) -> String {
    let mut prompt = String::from("<|im_start|>User\n");
    prompt.push_str(&input);
    prompt.push_str(&String::from("<|im_end|> <|im_start|>Assistant\n"));
    prompt
    
}